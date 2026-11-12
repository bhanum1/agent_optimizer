"""TuRBO-1 style trust-region BO with restarts (minimization, continuous box).

Each run: n_init Sobol points, then GP (ARD Matern-5/2) Thompson sampling inside a
trust region around the run's best point. The region grows on successes, shrinks on
failures; when it falls below length_min a new run starts. All history is kept;
the GP of a run is fit on that run's points (plus warm-start points near its center).
"""
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch


class Optimizer:
    max_ask = 1
    batch = False

    def __init__(self, space, seed=0, n_init=20, length_init=0.8, length_min=0.5 ** 7,
                 length_max=1.6, succ_tol=3, fail_tol=None, n_cand=5000, max_gp=400):
        self.lb = np.array([b[0] for b in space["bounds"]], float)
        self.ub = np.array([b[1] for b in space["bounds"]], float)
        self.dim = len(self.lb)
        self.seed = seed
        self.n_init = n_init
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.succ_tol = succ_tol
        self.fail_tol = fail_tol or max(4, self.dim)
        self.n_cand = n_cand
        self.max_gp = max_gp
        self.rng = np.random.default_rng(seed)
        self.X_all = np.zeros((0, self.dim))   # unit cube
        self.y_all = np.zeros(0)
        self.run_idx = []                      # indices into X_all for current run
        self.n_runs = 0
        self.init_queue = []
        self.pending = []
        self._new_run(first=True)

    # ---- helpers -------------------------------------------------------
    def _to_unit(self, X):
        return (np.asarray(X, float) - self.lb) / (self.ub - self.lb)

    def _from_unit(self, U):
        return self.lb + np.clip(U, 0, 1) * (self.ub - self.lb)

    def _new_run(self, first=False):
        from scipy.stats import qmc
        self.n_runs += 1
        self.length = self.length_init
        self.succ = 0
        self.fail = 0
        self.run_idx = []
        self.run_best = np.inf
        s = qmc.Sobol(self.dim, scramble=True, seed=int(self.rng.integers(2 ** 31)))
        self.init_queue = list(s.random(self.n_init))

    def _fit_gp(self, X, y):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.constraints import Interval
        from gpytorch.likelihoods import GaussianLikelihood
        Xt = torch.tensor(X, dtype=torch.double)
        mu, sd = y.mean(), y.std() if y.std() > 0 else 1.0
        yt = torch.tensor((y - mu) / sd, dtype=torch.double).unsqueeze(-1)
        lik = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim,
                                        lengthscale_constraint=Interval(0.005, 4.0)),
                           outputscale_constraint=Interval(0.05, 20.0))
        model = SingleTaskGP(Xt, yt, covar_module=kern, likelihood=lik)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        model.eval()
        return model

    def _propose(self):
        idx = np.array(self.run_idx)
        X = self.X_all[idx]
        y = self.y_all[idx]
        if len(idx) > self.max_gp:
            # keep the points closest to the run's best
            c = X[np.argmin(y)]
            keep = np.argsort(np.linalg.norm(X - c, axis=1))[: self.max_gp]
            X, y = X[keep], y[keep]
        model = self._fit_gp(X, y)
        center = X[np.argmin(y)]
        ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        w = ls / ls.mean()
        w = w / np.prod(w ** (1.0 / self.dim))
        lo = np.clip(center - w * self.length / 2, 0, 1)
        hi = np.clip(center + w * self.length / 2, 0, 1)
        from scipy.stats import qmc
        s = qmc.Sobol(self.dim, scramble=True, seed=int(self.rng.integers(2 ** 31)))
        pert = lo + (hi - lo) * s.random(self.n_cand)
        prob = min(20.0 / self.dim, 1.0)
        mask = self.rng.random((self.n_cand, self.dim)) <= prob
        none = np.where(mask.sum(1) == 0)[0]
        mask[none, self.rng.integers(0, self.dim, len(none))] = True
        cand = np.where(mask, pert, center)
        with torch.no_grad():
            post = model.posterior(torch.tensor(cand, dtype=torch.double))
            torch.manual_seed(int(self.rng.integers(2 ** 31)))
            samp = post.rsample(torch.Size([1])).squeeze().numpy()
        return cand[int(np.argmin(samp))]

    # ---- interface -----------------------------------------------------
    def ask(self, n):
        out = []
        for _ in range(n):
            if self.init_queue:
                u = self.init_queue.pop(0)
            elif len(self.run_idx) < 2:
                u = self.rng.random(self.dim)
            else:
                u = self._propose()
            self.pending.append(u)
            out.append([float(v) for v in self._from_unit(u)])
        return out

    def tell(self, X, y, g):
        U = self._to_unit(X)
        y = np.asarray(y, float)
        for u, v in zip(U, y):
            # match against outstanding asks (in unit space)
            hit = None
            for k, p in enumerate(self.pending):
                if np.allclose(p, u, atol=1e-9):
                    hit = k
                    break
            if hit is not None:
                self.pending.pop(hit)
            self.X_all = np.vstack([self.X_all, u])
            self.y_all = np.append(self.y_all, v)
            i = len(self.y_all) - 1
            in_init = len(self.run_idx) < self.n_init
            self.run_idx.append(i)
            if hit is None or in_init:
                self.run_best = min(self.run_best, v)
                continue
            # trust-region update after an acquisition point
            if v < self.run_best - 1e-3 * abs(self.run_best):
                self.succ += 1
                self.fail = 0
            else:
                self.succ = 0
                self.fail += 1
            self.run_best = min(self.run_best, v)
            if self.succ >= self.succ_tol:
                self.length = min(2 * self.length, self.length_max)
                self.succ = 0
            elif self.fail >= self.fail_tol:
                self.length /= 2
                self.fail = 0
            if self.length < self.length_min:
                self._new_run()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
