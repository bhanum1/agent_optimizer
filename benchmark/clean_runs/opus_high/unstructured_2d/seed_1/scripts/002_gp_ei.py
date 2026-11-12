"""GP + LogEI Bayesian optimizer (botorch), refit from all data on every ask.

y is warped with log(y - min + eps) to tame heavy upper tails; Matern-5/2 ARD GP
on inputs scaled to the unit cube. Acquisition optimized by optimize_acqf with
restarts. Every `explore_every`-th ask (if > 0) is a random point instead.
"""
import json
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_init=0, warp=True, explore_every=0,
                 num_restarts=20, raw_samples=1024):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.dim = int(space["dim"])
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.warp = bool(warp)
        self.explore_every = int(explore_every)
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)
        self.X, self.y = [], []
        self.pending = []
        self.n_asked = 0
        self.rng = np.random.default_rng(self.seed)

    def _to_unit(self, X):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(X, float) - lo) / (hi - lo)

    def _from_unit(self, U):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.clip(U, 0, 1) * (hi - lo)

    def _random(self):
        return self._from_unit(self.rng.random(self.dim)).tolist()

    def _bo(self):
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        X = torch.tensor(self._to_unit(self.X))
        y = np.asarray(self.y, float)
        if self.warp:
            span = max(np.std(y), 1e-9)
            y = np.log(y - y.min() + 0.05 * span)
        Y = torch.tensor(y).unsqueeze(-1)
        torch.manual_seed(int(self.rng.integers(2**31)))
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        acq = LogExpectedImprovement(model, best_f=Y.min(), maximize=False)
        b = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
        cand, _ = optimize_acqf(acq, bounds=b, q=1, num_restarts=self.num_restarts,
                                raw_samples=self.raw_samples)
        u = cand.detach().numpy()[0]
        # avoid exact repeats of measured points
        if np.min(np.linalg.norm(self._to_unit(self.X) - u, axis=1)) < 1e-6:
            u = np.clip(u + self.rng.normal(0, 1e-3, self.dim), 0, 1)
        return self._from_unit(u).tolist()

    def ask(self, n):
        out = []
        for _ in range(n):
            self.n_asked += 1
            if (len(self.y) < max(self.n_init, 2) or
                    (self.explore_every > 0 and self.n_asked % self.explore_every == 0)):
                p = self._random()
            else:
                p = self._bo()
            self.pending.append(p)
            out.append(p)
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append([float(v) for v in xi])
            self.y.append(float(yi))
            for j, p in enumerate(self.pending):
                if np.allclose(p, xi):
                    self.pending.pop(j)
                    break

    def save(self, path):
        d = {k: getattr(self, k) for k in
             ["dim", "seed", "n_init", "warp", "explore_every", "num_restarts",
              "raw_samples", "X", "y", "pending", "n_asked"]}
        d["bounds"] = self.bounds.tolist()
        d["rng"] = self.rng.bit_generator.state
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls.__new__(cls)
        for k, v in d.items():
            if k not in ("bounds", "rng"):
                setattr(o, k, v)
        o.bounds = np.array(d["bounds"], float)
        o.rng = np.random.default_rng()
        o.rng.bit_generator.state = d["rng"]
        return o
