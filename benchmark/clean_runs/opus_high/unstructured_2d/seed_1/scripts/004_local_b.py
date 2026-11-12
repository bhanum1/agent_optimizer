"""Local trust-region GP + LogEI (botorch) around a chosen center.

The GP (Matern-5/2 ARD, log-warped y) is fit on the measured points nearest the
center (those within `fit_radius`, at least `min_fit` nearest). EI is maximized in a
box of half-width `radius` around the best point seen inside the region (initially
`center`). The box shrinks by `shrink` after an ask that does not improve the local
best and grows back (up to the initial radius) after one that does.
"""
import json
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, center, radius=0.4, fit_radius=1.2, min_fit=10,
                 shrink=0.7, min_radius=0.02, seed=0, num_restarts=20,
                 raw_samples=1024):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.dim = int(space["dim"])
        self.center = [float(c) for c in center]
        self.radius0 = float(radius)
        self.radius = float(radius)
        self.fit_radius = float(fit_radius)
        self.min_fit = int(min_fit)
        self.shrink = float(shrink)
        self.min_radius = float(min_radius)
        self.seed = int(seed)
        self.num_restarts = int(num_restarts)
        self.raw_samples = int(raw_samples)
        self.X, self.y = [], []
        self.pending = []
        self.local_best = None
        self.rng = np.random.default_rng(self.seed)

    def _region_best(self):
        X = np.asarray(self.X, float)
        y = np.asarray(self.y, float)
        d = np.max(np.abs(X - np.asarray(self.center)), axis=1)
        m = d <= self.radius0 + 1e-9
        if not m.any():
            return np.asarray(self.center), np.inf
        i = np.flatnonzero(m)[np.argmin(y[m])]
        return X[i], y[i]

    def _bo(self):
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        Xa = np.asarray(self.X, float)
        ya = np.asarray(self.y, float)
        c, _ = self._region_best()
        d = np.linalg.norm(Xa - c, axis=1)
        idx = np.flatnonzero(d <= self.fit_radius)
        if len(idx) < self.min_fit:
            idx = np.argsort(d)[:self.min_fit]
        Xf, yf = Xa[idx], ya[idx]
        lo = np.maximum(c - self.radius, self.bounds[:, 0])
        hi = np.minimum(c + self.radius, self.bounds[:, 1])
        flo = np.minimum(Xf.min(0), lo)
        fhi = np.maximum(Xf.max(0), hi)
        scale = np.maximum(fhi - flo, 1e-9)
        X = torch.tensor((Xf - flo) / scale)
        span = max(np.std(yf), 1e-9)
        Y = torch.tensor(np.log(yf - yf.min() + 0.05 * span)).unsqueeze(-1)
        torch.manual_seed(int(self.rng.integers(2**31)))
        model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        acq = LogExpectedImprovement(model, best_f=Y.min(), maximize=False)
        b = torch.tensor(np.stack([(lo - flo) / scale, (hi - flo) / scale]))
        cand, _ = optimize_acqf(acq, bounds=b, q=1, num_restarts=self.num_restarts,
                                raw_samples=self.raw_samples)
        p = flo + cand.detach().numpy()[0] * scale
        if np.min(np.linalg.norm(Xa - p, axis=1)) < 1e-6:
            p = p + self.rng.normal(0, 1e-3 * self.radius, self.dim)
        return np.clip(p, lo, hi).tolist()

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.y) < 2:
                p = self.center
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
        _, b = self._region_best()
        if self.local_best is not None and len(X) == 1:
            if b < self.local_best - 1e-12:
                self.radius = min(self.radius0, self.radius / self.shrink)
            else:
                self.radius = max(self.min_radius, self.radius * self.shrink)
        self.local_best = float(b) if np.isfinite(b) else None

    def save(self, path):
        d = {k: getattr(self, k) for k in
             ["dim", "center", "radius0", "radius", "fit_radius", "min_fit", "shrink",
              "min_radius", "seed", "num_restarts", "raw_samples", "X", "y",
              "pending", "local_best"]}
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
