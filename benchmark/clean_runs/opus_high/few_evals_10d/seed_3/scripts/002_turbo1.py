"""TuRBO-1 style trust-region BO with a noise-aware GP (botorch).

Each ask: fit a Matern-5/2 ARD GP (learned noise) on all data in [0,1]^d with
standardized y; centre the trust region on the observed point with the lowest
posterior mean; maximize qLogNoisyExpectedImprovement inside the region.
Region grows after `succ_tol` successes, shrinks after `fail_tol` failures,
and restarts (full box) when it falls below `length_min`.
"""
import json
import math

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, length_init=0.8, length_min=0.5 ** 7,
                 length_max=1.6, succ_tol=3, fail_tol=None, n_restarts=8,
                 raw_samples=512):
        self.lo = [b[0] for b in space["bounds"]]
        self.hi = [b[1] for b in space["bounds"]]
        self.d = len(self.lo)
        self.seed = seed
        self.length_init = length_init
        self.length = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.succ_tol = succ_tol
        self.fail_tol = fail_tol if fail_tol is not None else max(4, self.d // 2)
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.succ = 0
        self.fail = 0
        self.X = []  # unit cube
        self.y = []
        self.n_asked = 0
        self.restart_from = 0  # index into data where the current TR run starts
        self.pending = []

    # --- helpers
    def _to_unit(self, x):
        return [(v - l) / (h - l) for v, l, h in zip(x, self.lo, self.hi)]

    def _from_unit(self, u):
        return [l + min(max(v, 0.0), 1.0) * (h - l) for v, l, h in zip(u, self.lo, self.hi)]

    def _fit(self):
        X = torch.tensor(self.X)
        y = torch.tensor(self.y).unsqueeze(-1)
        model = SingleTaskGP(X, y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        return model

    def ask(self, n):
        out = []
        for _ in range(n):
            torch.manual_seed(self.seed * 100003 + self.n_asked)
            np.random.seed((self.seed * 100003 + self.n_asked) % (2 ** 32))
            self.n_asked += 1
            if len(self.X) < 2:
                u = np.random.rand(self.d).tolist()
            else:
                u = self._propose()
            self.pending.append(u)
            out.append(self._from_unit(u))
        return out

    def _propose(self):
        model = self._fit()
        X = torch.tensor(self.X)
        with torch.no_grad():
            mu = model.posterior(X).mean.squeeze(-1)
        # centre: best posterior mean among points of the current TR run (or all if few)
        idx = list(range(self.restart_from, len(self.X)))
        if len(idx) == 0:
            idx = list(range(len(self.X)))
        centre = X[idx][mu[idx].argmin()]
        ls = model.covar_module.lengthscale.detach().squeeze()
        if ls.dim() == 0:
            ls = ls.repeat(self.d)
        w = ls / ls.mean()
        w = w / torch.prod(w.pow(1.0 / self.d))
        lb = torch.clamp(centre - w * self.length / 2, 0.0, 1.0)
        ub = torch.clamp(centre + w * self.length / 2, 0.0, 1.0)
        acq = qLogNoisyExpectedImprovement(model, X_baseline=X, prune_baseline=True)
        try:
            cand, _ = optimize_acqf(acq, bounds=torch.stack([lb, ub]), q=1,
                                    num_restarts=self.n_restarts,
                                    raw_samples=self.raw_samples)
            u = cand.squeeze(0).tolist()
        except Exception:
            u = (lb + (ub - lb) * torch.rand(self.d)).tolist()
        return u

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            u = self._to_unit(x)
            v = float(v)
            if not math.isfinite(v):
                v = max(self.y) if self.y else 0.0
            ours = False
            for i, p in enumerate(self.pending):
                if max(abs(a - b) for a, b in zip(p, u)) < 1e-6:
                    self.pending.pop(i)
                    ours = True
                    break
            if ours and self.X:
                prev_best = min(self.y[self.restart_from:]) if len(self.y) > self.restart_from else min(self.y)
                if v < prev_best - 1e-3 * abs(prev_best):
                    self.succ += 1
                    self.fail = 0
                else:
                    self.succ = 0
                    self.fail += 1
            self.X.append(u)
            self.y.append(v)
            if ours:
                if self.succ >= self.succ_tol:
                    self.length = min(2 * self.length, self.length_max)
                    self.succ = 0
                elif self.fail >= self.fail_tol:
                    self.length /= 2
                    self.fail = 0
                if self.length < self.length_min:
                    self.length = self.length_init
                    self.restart_from = len(self.X)
                    self.succ = self.fail = 0

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path) as f:
            obj.__dict__.update(json.load(f))
        return obj
