"""GP Bayesian optimization (botorch): SingleTaskGP with learned noise, qLogNEI.

Unconstrained, continuous box. State is just the observed data plus pending points
and an RNG counter, so save/load is exact up to GP refitting.
"""
import json
import math
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_init=10, local_frac=0.5, raw_samples=512,
                 num_restarts=8, **_):
        self.bounds = [list(b) for b in space["bounds"]]
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.local_frac = float(local_frac)
        self.raw_samples = int(raw_samples)
        self.num_restarts = int(num_restarts)
        self.X, self.y = [], []
        self.pending = []
        self.n_asked = 0

    def _rng(self):
        return np.random.default_rng(self.seed * 100003 + self.n_asked)

    def ask(self, n):
        out = []
        for _ in range(n):
            out.append(self._ask_one())
            self.n_asked += 1
        self.pending.extend(out)
        return out

    def _ask_one(self):
        b = np.array(self.bounds, dtype=float)
        lo, hi = b[:, 0], b[:, 1]
        rng = self._rng()
        if len(self.y) < self.n_init:
            return list(lo + rng.random(len(lo)) * (hi - lo))
        torch.manual_seed(self.seed * 7919 + self.n_asked)
        X = torch.tensor(self.X)
        Y = torch.tensor(self.y).unsqueeze(-1)
        bt = torch.tensor(b.T)
        model = SingleTaskGP(X, Y, input_transform=Normalize(X.shape[-1], bounds=bt),
                             outcome_transform=Standardize(1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        Xp = torch.tensor(self.pending) if self.pending else None
        acq = qLogNoisyExpectedImprovement(model, X_baseline=X, X_pending=Xp,
                                           sampler=SobolQMCNormalSampler(torch.Size([256])),
                                           prune_baseline=True)
        # Local search box around the posterior-mean-best point, alternating with global.
        use_local = rng.random() < self.local_frac
        if use_local:
            with torch.no_grad():
                mu = model.posterior(X).mean.squeeze(-1)
            xb = X[int(torch.argmin(mu))].numpy()
            ls = model.covar_module.lengthscale.detach().numpy().ravel()
            ls = ls / np.exp(np.mean(np.log(ls)))
            w = 0.2 * (hi - lo) * np.clip(ls, 0.25, 4.0)
            blo, bhi = np.maximum(lo, xb - w), np.minimum(hi, xb + w)
            sb = torch.tensor(np.stack([blo, bhi]))
        else:
            sb = bt
        cand, _ = optimize_acqf(acq, bounds=sb, q=1, num_restarts=self.num_restarts,
                                raw_samples=self.raw_samples)
        x = cand.detach().numpy().ravel()
        return list(np.clip(x, lo, hi))

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            xi = [float(v) for v in xi]
            self.X.append(xi)
            self.y.append(float(yi))
            for k, p in enumerate(self.pending):
                if np.allclose(p, xi):
                    self.pending.pop(k)
                    break

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path) as f:
            o.__dict__.update(json.load(f))
        return o
