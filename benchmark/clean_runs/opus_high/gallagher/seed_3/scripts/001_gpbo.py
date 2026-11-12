"""GP Bayesian optimization: scrambled Sobol initial design, then LogEI on a
Matern-5/2 ARD GP fitted to standardized y. Deterministic objective, minimized.
State is the told data, outstanding asks and RNG counter, saved as JSON."""
import json
import math

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_init=10, **_):
        self.bounds = [list(map(float, b)) for b in space["bounds"]]
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.X, self.y = [], []
        self.pending = []
        self.n_asked = 0

    # unit cube <-> bounds
    def _to_unit(self, X):
        lo, hi = np.array(self.bounds).T
        return (np.asarray(X, float) - lo) / (hi - lo)

    def _from_unit(self, U):
        lo, hi = np.array(self.bounds).T
        return lo + np.asarray(U, float) * (hi - lo)

    def _init_point(self, k):
        s = qmc.Sobol(d=len(self.bounds), scramble=True, seed=self.seed)
        pts = s.random(2 ** max(1, math.ceil(math.log2(max(self.n_init, k + 1)))))
        return self._from_unit(pts[k]).tolist()

    def _bo_point(self):
        X = torch.tensor(self._to_unit(self.X + self.pending))
        y = np.array(self.y, float)
        y = np.where(np.isfinite(y), y, np.nanmax(y[np.isfinite(y)]) if np.isfinite(y).any() else 0.0)
        Y = torch.tensor(-y).unsqueeze(-1)  # botorch maximizes
        X = X[: len(self.y)]
        torch.manual_seed(self.seed * 1000 + self.n_asked)
        gp = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        acq = LogExpectedImprovement(gp, best_f=Y.max())
        d = X.shape[1]
        b = torch.stack([torch.zeros(d), torch.ones(d)])
        cand, _ = optimize_acqf(acq, bounds=b, q=1, num_restarts=10, raw_samples=512)
        u = cand.detach().numpy()[0]
        # avoid exact repeats of known points
        if len(X) and np.min(np.linalg.norm(X.numpy() - u, axis=1)) < 1e-6:
            u = np.random.default_rng(self.seed + self.n_asked).random(d)
        return self._from_unit(np.clip(u, 0, 1)).tolist()

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.y) + len(self.pending) < self.n_init:
                x = self._init_point(len(self.y) + len(self.pending))
            else:
                x = self._bo_point()
            self.pending.append(x)
            self.n_asked += 1
            out.append(x)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = [float(a) for a in x]
            for i, p in enumerate(self.pending):
                if np.allclose(p, x):
                    self.pending.pop(i)
                    break
            self.X.append(x)
            self.y.append(float(v) if v is not None and math.isfinite(float(v)) else float("nan"))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path) as f:
            o.__dict__.update(json.load(f))
        return o
