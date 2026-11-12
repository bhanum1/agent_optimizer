"""TuRBO-1 style trust-region BO: GP on log-shifted, standardized y; Thompson sampling in the TR."""
import pickle
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_init=10, length_init=0.8, length_min=0.5 ** 7,
                 length_max=1.6, succ_tol=3, fail_tol=5, n_local=250, n_cand=4096):
        b = np.array(space["bounds"], dtype=float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.d = len(self.lo)
        self.seed, self.n_init = seed, n_init
        self.length_init, self.length_min, self.length_max = length_init, length_min, length_max
        self.succ_tol, self.fail_tol = succ_tol, fail_tol
        self.n_local, self.n_cand = n_local, n_cand
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((0, self.d)); self.y = np.zeros(0)
        self.pending = []          # unit-cube points from restart design not yet asked
        self.length = length_init
        self.succ = self.fail = 0
        self.tr_best = np.inf      # best value seen within the current TR
        self.tr_start = 0          # index in X where current TR began
        self.restarts = 0
        self.init_left = 0         # restart-design points still to be told

    # ---- helpers
    def _unit(self, X): return (np.asarray(X) - self.lo) / (self.hi - self.lo)
    def _orig(self, U): return self.lo + np.asarray(U) * (self.hi - self.lo)

    def _transform(self, y):
        y = np.asarray(y, float)
        ymin, med = y.min(), np.median(y)
        delta = max(1e-3 * (med - ymin), 1e-12)
        z = np.log(y - ymin + delta)
        return (z - z.mean()) / (z.std() + 1e-12)

    def _restart(self):
        self.restarts += 1
        s = qmc.Sobol(self.d, scramble=True, seed=int(self.rng.integers(2 ** 31)))
        self.pending = list(s.random(self.n_init))
        self.init_left = self.n_init
        self.length = self.length_init
        self.succ = self.fail = 0
        self.tr_best = np.inf
        self.tr_start = len(self.y)

    def _center(self):
        y_tr = self.y[self.tr_start:]
        if len(y_tr) == 0:
            i = int(np.argmin(self.y))
        else:
            i = self.tr_start + int(np.argmin(y_tr))
        return self._unit(self.X[i])

    def _propose(self):
        U = self._unit(self.X)
        c = self._center()
        dist = np.linalg.norm(U - c, axis=1)
        idx = np.argsort(dist)[: self.n_local]
        Ut, yt = U[idx], self._transform(self.y[idx])
        Xt = torch.tensor(Ut); Yt = torch.tensor(yt).unsqueeze(-1)
        torch.manual_seed(int(self.rng.integers(2 ** 31)))
        model = SingleTaskGP(Xt, Yt)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        ck = model.covar_module
        ls = (ck.lengthscale if hasattr(ck, "lengthscale") else ck.base_kernel.lengthscale)
        ls = ls.detach().numpy().ravel()
        w = ls / ls.mean(); w = w / np.prod(w) ** (1.0 / self.d)
        lb = np.clip(c - w * self.length / 2, 0, 1); ub = np.clip(c + w * self.length / 2, 0, 1)
        s = qmc.Sobol(self.d, scramble=True, seed=int(self.rng.integers(2 ** 31)))
        pert = lb + (ub - lb) * s.random(self.n_cand)
        prob = min(20.0 / self.d, 1.0)
        mask = self.rng.random((self.n_cand, self.d)) <= prob
        none = np.where(mask.sum(1) == 0)[0]
        mask[none, self.rng.integers(0, self.d, len(none))] = True
        cand = np.where(mask, pert, c)
        model.eval()
        with torch.no_grad():
            post = model.posterior(torch.tensor(cand))
            f = post.rsample(torch.Size([1])).squeeze().numpy()
        # avoid exact duplicates
        return cand[int(np.argmin(f))]

    # ---- API
    def ask(self, n):
        out = []
        for _ in range(n):
            if self.pending:
                out.append(self._orig(self.pending.pop(0)).tolist())
            elif len(self.y) < 2:
                self._restart(); out.append(self._orig(self.pending.pop(0)).tolist())
            else:
                out.append(self._orig(self._propose()).tolist())
        return out

    def tell(self, X, y, g=None):
        X = np.asarray(X, float).reshape(-1, self.d); y = np.asarray(y, float)
        warm = len(y) > 1
        for xi, yi in zip(X, y):
            self.X = np.vstack([self.X, xi]); self.y = np.append(self.y, yi)
            if warm:
                continue
            if self.init_left > 0:
                self.init_left -= 1
                self.tr_best = min(self.tr_best, yi)
                continue
            if yi < self.tr_best - 1e-3 * abs(self.tr_best):
                self.succ += 1; self.fail = 0
            else:
                self.fail += 1; self.succ = 0
            self.tr_best = min(self.tr_best, yi)
            if self.succ >= self.succ_tol:
                self.length = min(2 * self.length, self.length_max); self.succ = 0
            elif self.fail >= self.fail_tol:
                self.length /= 2; self.fail = 0
            if self.length < self.length_min:
                self._restart()
        if warm:
            self.tr_best = min(self.tr_best, float(self.y[self.tr_start:].min()))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
