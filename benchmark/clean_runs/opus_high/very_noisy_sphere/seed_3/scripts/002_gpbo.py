"""GP Bayesian optimization for a noisy objective with heavy-tailed values.

y is warped to normal scores of its ranks (copula) before fitting, so large
outliers do not dominate. Acquisition: LogEI with the best posterior mean at
observed points as incumbent (a cheap noisy-EI). A fraction `local_frac` of
asks restricts the search to a box of half-width `local_hw` around that incumbent.
"""
import json
import pickle
import numpy as np
import torch
from scipy.stats import norm


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_init=10, local_frac=0.3, local_hw=0.1, raw=512, restarts=8):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.dim = space["dim"]
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.n_init = n_init
        self.local_frac = local_frac
        self.local_hw = local_hw
        self.raw = raw
        self.restarts = restarts
        self.X, self.y = [], []
        self.n_asked = 0

    def _fit(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        X = (np.array(self.X) - lo) / (hi - lo)
        y = np.array(self.y)
        r = np.argsort(np.argsort(y))
        z = norm.ppf((r + 0.5) / len(y))
        z = (z - z.mean()) / z.std()
        Xt = torch.tensor(X, dtype=torch.double)
        zt = torch.tensor(z, dtype=torch.double).unsqueeze(-1)
        gp = SingleTaskGP(Xt, zt)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        return gp, Xt

    def ask(self, n):
        out = []
        for _ in range(n):
            self.n_asked += 1
            if len(self.y) < self.n_init:
                lo, hi = self.bounds[:, 0], self.bounds[:, 1]
                out.append(list(lo + self.rng.random(self.dim) * (hi - lo)))
                continue
            from botorch.acquisition import LogExpectedImprovement
            from botorch.optim import optimize_acqf
            torch.manual_seed(int(self.rng.integers(2**31)))
            gp, Xt = self._fit()
            with torch.no_grad():
                mu = gp.posterior(Xt).mean.squeeze(-1)
            i = int(torch.argmin(mu))
            # minimize -> maximize negative
            from botorch.acquisition.objective import ScalarizedPosteriorTransform
            pt = ScalarizedPosteriorTransform(weights=torch.tensor([-1.0], dtype=torch.double))
            acq = LogExpectedImprovement(gp, best_f=-mu[i].item(), posterior_transform=pt)
            b = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.double)
            if self.rng.random() < self.local_frac:
                c = Xt[i]
                b = torch.stack([(c - self.local_hw).clamp(0, 1), (c + self.local_hw).clamp(0, 1)])
            x, _ = optimize_acqf(acq, bounds=b, q=1, num_restarts=self.restarts, raw_samples=self.raw)
            u = x.squeeze(0).detach().numpy().clip(0, 1)
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            out.append([float(v) for v in lo + u * (hi - lo)])
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append([float(v) for v in xi])
            self.y.append(float(yi))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
