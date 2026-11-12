"""Local trust-region refinement with a quadratic model, for a deterministic
objective (minimized) in user units (all-continuous linear space).
Centre = best point told so far. Points within fit_radius of the centre are fitted with a
weighted full quadratic; while fewer than 6 are available, axis probes at
centre +/- delta are asked instead. The model minimum within the trust region
(radius delta, box around centre) is asked; delta expands on improvement and
shrinks otherwise. Candidates closer than min_sep to a measured point are moved.
Warm-start friendly. State saved as JSON."""
import json

import numpy as np
from scipy.optimize import minimize


def _feats(D):
    D = np.atleast_2d(D)
    a, b = D[:, 0], D[:, 1]
    return np.column_stack([np.ones(len(D)), a, b, a * a, a * b, b * b])


class Optimizer:
    max_ask = 1

    def __init__(self, space, delta=0.1, fit_radius=0.6, min_sep=0.01,
                 delta_min=0.005, delta_max=0.4, **_):
        self.bounds = [list(map(float, b)) for b in space["bounds"]]
        self.delta = float(delta)
        self.fit_radius = float(fit_radius)
        self.min_sep = float(min_sep)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.X, self.y = [], []
        self.last_best = None

    def _clip(self, p):
        lo, hi = np.array(self.bounds).T
        return np.clip(p, lo, hi)

    def _far(self, p):
        X = np.array(self.X)
        return len(X) == 0 or np.min(np.linalg.norm(X - p, axis=1)) >= self.min_sep

    def _propose(self):
        X, y = np.array(self.X), np.array(self.y)
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        c = X[np.argmin(y)]
        d = np.linalg.norm(X - c, axis=1)
        near = d <= self.fit_radius
        if near.sum() < 6:
            for e in ([1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]):
                p = self._clip(c + self.delta * np.array(e, float))
                if self._far(p):
                    return p
        D, v = X[near] - c, y[near]
        w = np.exp(-(d[near] / self.fit_radius) ** 2)
        F = _feats(D)
        coef, *_ = np.linalg.lstsq(F * w[:, None], v * w, rcond=None)
        model = lambda s: float((_feats(s) @ coef)[0])
        lo = np.maximum(np.array(self.bounds)[:, 0] - c, -self.delta)
        hi = np.minimum(np.array(self.bounds)[:, 1] - c, self.delta)
        best, bval = None, np.inf
        starts = [np.zeros(2)] + [np.array([sx, sy]) * self.delta * 0.7
                                  for sx in (-1, 1) for sy in (-1, 1)]
        for s0 in starts:
            r = minimize(model, s0, bounds=list(zip(lo, hi)), method="L-BFGS-B")
            if r.fun < bval and self._far(c + r.x):
                best, bval = r.x, r.fun
        if best is None or np.linalg.norm(best) < self.min_sep:
            g = coef[1:3]
            dirn = -g / (np.linalg.norm(g) + 1e-12)
            best = dirn * max(self.delta, 2 * self.min_sep)
        return self._clip(c + best)

    def ask(self, n):
        return [self._propose().tolist()]

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            self.X.append(list(map(float, x)))
            self.y.append(float(v) if v is not None and np.isfinite(v) else float("inf"))
        cur = min(self.y)
        if self.last_best is not None and len(X) == 1:
            if cur < self.last_best - 1e-9:
                self.delta = min(self.delta * 1.5, self.delta_max)
            else:
                self.delta = max(self.delta * 0.5, self.delta_min)
        self.last_best = cur

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path) as f:
            o.__dict__.update(json.load(f))
        return o
