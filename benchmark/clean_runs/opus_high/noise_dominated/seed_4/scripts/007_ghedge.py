"""Global Thompson-sampling hedge that excludes a known basin.

A GP on log-warped y is fitted only to points farther than `fit_r` (user units) from
`center`, so the warp and lengthscales describe the landscape away from the known basin.
Each step draws one joint posterior sample over a candidate set (scrambled Sobol over the
box plus Gaussian perturbations of the best fitted points) restricted to distance > `excl_r`
from `center`, and asks the candidate where the sample is smallest.
"""
import json
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, center, seed=0, excl_r=2.5, fit_r=1.5, n_sobol=2000,
                 n_local=800, local_sd=0.04, n_top=20):
        b = np.array(space["bounds"], dtype=float)
        self.lo, self.hi = b[:, 0].tolist(), b[:, 1].tolist()
        self.center = [float(v) for v in center]
        self.seed = int(seed)
        self.excl_r = float(excl_r)
        self.fit_r = float(fit_r)
        self.n_sobol = int(n_sobol)
        self.n_local = int(n_local)
        self.local_sd = float(local_sd)
        self.n_top = int(n_top)
        self.X, self.y = [], []
        self.n_asked = 0

    def _unit(self, X):
        lo, hi = np.array(self.lo), np.array(self.hi)
        return (np.asarray(X, float) - lo) / (hi - lo)

    def _from_unit(self, U):
        lo, hi = np.array(self.lo), np.array(self.hi)
        return lo + np.clip(U, 0, 1) * (hi - lo)

    @staticmethod
    def _warp(y):
        q25, q50 = np.percentile(y, [25, 50])
        s = max(q50 - q25, 1e-9)
        z = np.log(y - y.min() + s)
        return (z - z.mean()) / (z.std() + 1e-12)

    def _dist(self, X):
        return np.linalg.norm(np.asarray(X, float) - np.array(self.center), axis=1)

    def _next(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from scipy.stats import qmc
        torch.manual_seed(self.seed * 100003 + self.n_asked)
        rng = np.random.default_rng(self.seed * 7919 + self.n_asked)
        X = np.array(self.X); y = np.array(self.y)
        keep = self._dist(X) > self.fit_r
        X, y = X[keep], y[keep]
        Xt = torch.tensor(self._unit(X)); zt = torch.tensor(self._warp(y)).unsqueeze(-1)
        m = SingleTaskGP(Xt, zt)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
        except Exception:
            pass
        d = Xt.shape[1]
        sob = qmc.Sobol(d, scramble=True, seed=int(rng.integers(2**31))).random(self.n_sobol)
        top = self._unit(X[np.argsort(y)[: self.n_top]])
        loc = top[rng.integers(len(top), size=self.n_local)] + rng.normal(0, self.local_sd, (self.n_local, d))
        C = np.clip(np.vstack([sob, loc]), 0, 1)
        C = C[self._dist(self._from_unit(C)) > self.excl_r]
        with torch.no_grad():
            f = m.posterior(torch.tensor(C)).rsample().squeeze()
        return self._from_unit(C[int(torch.argmin(f))]).tolist()

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.y) < 2:
                rng = np.random.default_rng(self.seed + self.n_asked)
                p = self._from_unit(rng.random(len(self.lo))).tolist()
            else:
                p = self._next()
            self.n_asked += 1
            out.append([float(v) for v in p])
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append([float(v) for v in xi]); self.y.append(float(yi))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path) as f:
            o.__dict__.update(json.load(f))
        return o
