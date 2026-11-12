"""Basin-focused trust-region BO for a deterministic box with flat plateaus.

Like TuRBO-1 (Matern-5/2 ARD GP on log-shifted y, Thompson sampling over Sobol
candidates in an ARD-shaped box), but: the TR is always centered on the global best;
the GP uses the n_local points nearest it from ALL data and learns a noise term (the
plateaus act as noise); when L falls below L_min it re-expands to L_init instead of
restarting elsewhere; a fraction p_wide of asks samples a wide box (L_max).
"""
import pickle
import warnings
import numpy as np
import torch


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_init=10, L_init=0.4, L_min=0.03, L_max=1.0,
                 succ_tol=3, fail_tol=5, n_local=400, n_cand=5000, p_wide=0.15,
                 noise_max=0.3):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lo)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_init, self.L_init, self.L_min, self.L_max = n_init, L_init, L_min, L_max
        self.succ_tol, self.fail_tol = succ_tol, fail_tol
        self.n_local, self.n_cand = n_local, n_cand
        self.p_wide, self.noise_max = p_wide, noise_max
        self.n_reexpand = 0
        self.X = np.zeros((0, self.d))   # unit cube, all observations
        self.y = np.zeros(0)
        self.tr_idx = []                 # indices of observations belonging to current TR
        self.init_queue = []             # pending init points (unit) of a fresh TR
        self.L = L_init
        self.succ = self.fail = 0
        self.centers = []                # converged TR centers (unit) and values
        self.n_restarts = 0
        self.pending = []

    # --- helpers
    def _to_unit(self, x):
        return (np.asarray(x, float) - self.lo) / (self.hi - self.lo)

    def _from_unit(self, u):
        return self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)

    def _sobol(self, n):
        from scipy.stats import qmc
        warnings.filterwarnings("ignore")
        s = qmc.Sobol(self.d, scramble=True, seed=int(self.rng.integers(2 ** 31)))
        return s.random(n)

    def _start_tr(self):
        self.L = self.L_init
        self.succ = self.fail = 0
        self.tr_idx = []
        cand = self._sobol(512)
        if self.centers:
            C = np.array([c for c, _ in self.centers])
            dist = np.min(np.linalg.norm(cand[:, None] - C[None], axis=2), axis=1)
            cand = cand[np.argsort(-dist)[:4 * self.n_init]]
        idx = self.rng.choice(len(cand), self.n_init, replace=False)
        self.init_queue = [c for c in cand[idx]]

    def _fit_and_sample(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.constraints import Interval
        from gpytorch.likelihoods import GaussianLikelihood

        Xt, yt = self.X, self.y
        center = self.X[np.argmin(yt)]
        L = self.L_max if self.rng.random() < self.p_wide else self.L
        if len(yt) > self.n_local:
            dist = np.linalg.norm(Xt - center, axis=1)
            keep = np.argsort(dist)[:self.n_local]
            Xt, yt = Xt[keep], yt[keep]
        # log-shift transform to tame heavy tails, then standardize
        span = np.median(yt) - yt.min()
        z = np.log(yt - yt.min() + 0.05 * span + 1e-12)
        z = (z - z.mean()) / (z.std() + 1e-12)
        torch.manual_seed(int(self.rng.integers(2 ** 31)))
        tX = torch.tensor(Xt, dtype=torch.double)
        tz = torch.tensor(z, dtype=torch.double).unsqueeze(-1)
        lik = GaussianLikelihood(noise_constraint=Interval(1e-6, self.noise_max))
        kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.d,
                                        lengthscale_constraint=Interval(0.005, 4.0)),
                           outputscale_constraint=Interval(0.05, 20.0))
        gp = SingleTaskGP(tX, tz, covar_module=kern, likelihood=lik)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        ls = gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        w = ls / ls.mean()
        w = w / np.prod(w) ** (1.0 / self.d)
        lb = np.clip(center - w * L / 2, 0, 1)
        ub = np.clip(center + w * L / 2, 0, 1)
        cand = lb + (ub - lb) * self._sobol(self.n_cand)
        # perturb only a subset of dims for some candidates (TuRBO style)
        prob = min(20.0 / self.d, 1.0)
        mask = self.rng.random(cand.shape) <= prob
        cand = np.where(mask, cand, center)
        gp.eval()
        with torch.no_grad():
            post = gp.posterior(torch.tensor(cand, dtype=torch.double))
            f = post.rsample(torch.Size([1])).squeeze().numpy()
        return cand[int(np.argmin(f))]

    # --- interface
    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.tr_idx and not self.init_queue:
                self._start_tr()
            if self.init_queue:
                u = self.init_queue.pop(0)
            elif len(self.tr_idx) < 2:
                u = self._sobol(1)[0]
            else:
                u = self._fit_and_sample()
            x = self._from_unit(u).tolist()
            self.pending.append(x)
            out.append(x)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v)
            ours = x in self.pending
            if ours:
                self.pending.remove(x)
            u = self._to_unit(x)
            prev_best = self.y[self.tr_idx].min() if self.tr_idx else np.inf
            self.X = np.vstack([self.X, u])
            self.y = np.append(self.y, v)
            self.tr_idx.append(len(self.y) - 1)
            in_init = ours and (len(self.init_queue) > 0 or len(self.tr_idx) <= self.n_init)
            if not ours or in_init or not np.isfinite(prev_best):
                continue
            if v < prev_best - 1e-3 * abs(prev_best):
                self.succ += 1
                self.fail = 0
            else:
                self.fail += 1
                self.succ = 0
            if self.succ >= self.succ_tol:
                self.L = min(2 * self.L, self.L_max)
                self.succ = 0
            elif self.fail >= self.fail_tol:
                self.L /= 2
                self.fail = 0
            if self.L < self.L_min:
                self.L = self.L_init
                self.n_reexpand += 1

    def save(self, path):
        d = dict(self.__dict__)
        d["rng"] = self.rng.bit_generator.state
        with open(path, "wb") as fh:
            pickle.dump(d, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        obj = cls.__new__(cls)
        rs = d.pop("rng")
        obj.__dict__.update(d)
        obj.rng = np.random.default_rng()
        obj.rng.bit_generator.state = rs
        return obj
