"""Local noise-aware GP-BO on clipped y, in a trust box around the best posterior-mean point.

y is clipped above at min(y) + clip_range, so the model resolves the good region in raw units
and bad points / upward noise outliers only say "not good". Acquisition: qLogNEI inside a box
centred on the argmin of the posterior mean over observed points, shaped by the ARD
lengthscales, of geometric-mean side `tr_frac` (unit cube). Every `explore_every`-th point
(if > 0) searches the whole box instead.
"""
import json
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, clip_range=15.0, tr_frac=0.4, explore_every=0,
                 raw_samples=1024, num_restarts=10):
        b = np.array(space["bounds"], dtype=float)
        self.lo, self.hi = b[:, 0].tolist(), b[:, 1].tolist()
        self.seed = int(seed)
        self.clip_range = float(clip_range)
        self.tr_frac = float(tr_frac)
        self.explore_every = int(explore_every)
        self.raw_samples = int(raw_samples)
        self.num_restarts = int(num_restarts)
        self.X, self.y = [], []
        self.n_asked = 0

    def _unit(self, X):
        lo, hi = np.array(self.lo), np.array(self.hi)
        return (np.asarray(X, float) - lo) / (hi - lo)

    def _from_unit(self, U):
        lo, hi = np.array(self.lo), np.array(self.hi)
        return lo + np.clip(U, 0, 1) * (hi - lo)

    def _fit(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        X = np.array(self.X); y = np.array(self.y)
        z = np.minimum(y, y.min() + self.clip_range)
        z = (z - z.mean()) / (z.std() + 1e-12)
        Xt = torch.tensor(self._unit(X)); zt = torch.tensor(z).unsqueeze(-1)
        m = SingleTaskGP(Xt, zt)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
        except Exception:
            pass
        return m, Xt

    def _next(self):
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.acquisition.objective import LinearMCObjective
        from botorch.optim import optimize_acqf
        torch.manual_seed(self.seed * 100003 + self.n_asked)
        m, Xt = self._fit()
        d = Xt.shape[1]
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        explore = self.explore_every > 0 and (self.n_asked + 1) % self.explore_every == 0
        if not explore:
            with torch.no_grad():
                mu = m.posterior(Xt).mean.squeeze(-1)
            c = Xt[int(torch.argmin(mu))]
            ls = m.covar_module.lengthscale.detach().squeeze()
            w = ls / ls.prod() ** (1.0 / d) * self.tr_frac
            bounds = torch.stack([(c - w / 2).clamp(0, 1), (c + w / 2).clamp(0, 1)])
        obj = LinearMCObjective(torch.tensor([-1.0]))
        acq = qLogNoisyExpectedImprovement(m, X_baseline=Xt, objective=obj, prune_baseline=True)
        cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=self.num_restarts,
                                raw_samples=self.raw_samples)
        return self._from_unit(cand.detach().numpy()[0]).tolist()

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
