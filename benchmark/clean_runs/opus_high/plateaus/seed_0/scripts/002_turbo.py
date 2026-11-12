import json

import numpy as np
import torch
from scipy.stats import qmc


class Optimizer:
    """TuRBO-1 style trust-region BO: local GP (Matern-5/2 ARD) + Thompson sampling.

    Works in the unit cube. On TR collapse it restarts with a fresh Sobol design,
    and the new TR's GP only uses points from the current restart (as in TuRBO).
    """

    def __init__(self, space, seed=0, n_init=20, length_init=0.8, length_min=0.0078125,
                 length_max=1.6, succ_tol=3, fail_tol=None, n_cand=2048, max_gp=300,
                 use_warm_for_tr=True):
        self.lb = np.array(space["bounds"], dtype=float)[:, 0]
        self.ub = np.array(space["bounds"], dtype=float)[:, 1]
        self.d = len(self.lb)
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.length_init = float(length_init)
        self.length_min = float(length_min)
        self.length_max = float(length_max)
        self.succ_tol = int(succ_tol)
        self.fail_tol = int(fail_tol) if fail_tol is not None else max(4, self.d)
        self.n_cand = int(n_cand)
        self.max_gp = int(max_gp)
        self.use_warm_for_tr = bool(use_warm_for_tr)
        self.rng = np.random.default_rng(self.seed)
        self.X = []          # unit-cube points, all told
        self.y = []
        self.tr_start = 0    # index into X where the current TR's data begins
        self.length = self.length_init
        self.succ = 0
        self.fail = 0
        self.n_restarts = 0
        self.init_queue = []  # unit points still to propose for current restart's design
        self.pending = []
        self.started = False

    # -- helpers
    def _to_unit(self, x):
        return (np.asarray(x, float) - self.lb) / (self.ub - self.lb)

    def _from_unit(self, u):
        return self.lb + np.asarray(u, float) * (self.ub - self.lb)

    def _new_design(self):
        s = qmc.Sobol(d=self.d, scramble=True, seed=int(self.rng.integers(2**31)))
        self.init_queue = s.random(self.n_init).tolist()

    def _restart(self):
        self.n_restarts += 1
        self.tr_start = len(self.X)
        self.length = self.length_init
        self.succ = self.fail = 0
        self._new_design()

    def _fit_and_sample(self, n):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.constraints import Interval
        from gpytorch.likelihoods import GaussianLikelihood

        X = np.array(self.X[self.tr_start:])
        y = np.array(self.y[self.tr_start:])
        ib = int(np.argmin(y))
        center = X[ib]
        if len(X) > self.max_gp:
            dist = np.max(np.abs(X - center), axis=1)
            idx = np.argsort(dist)[: self.max_gp]
            X, y = X[idx], y[idx]
        ys = (y - y.mean()) / (y.std() + 1e-12)
        torch.manual_seed(int(self.rng.integers(2**31)))
        tX = torch.tensor(X, dtype=torch.float64)
        ty = torch.tensor(ys, dtype=torch.float64).unsqueeze(-1)
        lik = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.d,
                                        lengthscale_constraint=Interval(0.005, 4.0)),
                           outputscale_constraint=Interval(0.05, 20.0))
        model = SingleTaskGP(tX, ty, likelihood=lik, covar_module=kern)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        w = ls / ls.mean()
        w = w / np.prod(w ** (1.0 / self.d))
        lo = np.clip(center - w * self.length / 2, 0, 1)
        hi = np.clip(center + w * self.length / 2, 0, 1)
        sob = qmc.Sobol(d=self.d, scramble=True, seed=int(self.rng.integers(2**31)))
        pert = lo + (hi - lo) * sob.random(self.n_cand)
        prob = min(20.0 / self.d, 1.0)
        mask = self.rng.random((self.n_cand, self.d)) <= prob
        none = np.where(mask.sum(1) == 0)[0]
        mask[none, self.rng.integers(0, self.d, size=len(none))] = True
        cand = np.where(mask, pert, center)
        model.eval()
        out = []
        with torch.no_grad():
            post = model.posterior(torch.tensor(cand, dtype=torch.float64))
            for _ in range(n):
                s = post.rsample().squeeze().numpy()
                j = int(np.argmin(s))
                out.append(cand[j].tolist())
                cand = np.delete(cand, j, axis=0)
                post = model.posterior(torch.tensor(cand, dtype=torch.float64))
        return out

    # -- interface
    def ask(self, n):
        if not self.started:
            self.started = True
            if not (self.use_warm_for_tr and len(self.X) >= max(2, self.d)):
                self._new_design()
        pts = []
        while len(pts) < n and self.init_queue:
            pts.append(self.init_queue.pop(0))
        if len(pts) < n and len(self.X) - self.tr_start < 2:
            self._new_design()
            while len(pts) < n and self.init_queue:
                pts.append(self.init_queue.pop(0))
        self.pending += [[p, False] for p in pts]
        if len(pts) < n:
            new = self._fit_and_sample(n - len(pts))
            self.pending += [[p, True] for p in new]
            pts += new
        return [self._from_unit(p).tolist() for p in pts]

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            u = self._to_unit(x).tolist()
            v = float(v)
            if not np.isfinite(v):
                v = max(self.y) * 2 if self.y else 1e10
            is_model = any(m and np.allclose(u, p, atol=1e-9) for p, m in self.pending)
            self.pending = [pm for pm in self.pending if not np.allclose(u, pm[0], atol=1e-9)]
            prev = self.y[self.tr_start:]
            best_prev = min(prev) if prev else np.inf
            self.X.append(u)
            self.y.append(v)
            # TR update only for model-proposed points
            if not is_model:
                continue
            if v < best_prev - 1e-3 * abs(best_prev):
                self.succ += 1
                self.fail = 0
            else:
                self.succ = 0
                self.fail += 1
            if self.succ == self.succ_tol:
                self.length = min(2.0 * self.length, self.length_max)
                self.succ = 0
            elif self.fail == self.fail_tol:
                self.length /= 2.0
                self.fail = 0
            if self.length < self.length_min:
                self._restart()

    def save(self, path):
        d = {k: v for k, v in self.__dict__.items() if k not in ("rng", "lb", "ub")}
        d["lb"], d["ub"] = self.lb.tolist(), self.ub.tolist()
        d["rng"] = self.rng.bit_generator.state
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls.__new__(cls)
        rs = d.pop("rng")
        o.__dict__.update(d)
        o.lb, o.ub = np.array(d["lb"]), np.array(d["ub"])
        o.rng = np.random.default_rng()
        o.rng.bit_generator.state = rs
        return o
