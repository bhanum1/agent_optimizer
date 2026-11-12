import pickle
import numpy as np


class Optimizer:
    """Local quadratic-model trust region. Each step fits a full quadratic (weighted
    least squares) to the points within `fit_radius` of the current best and proposes
    its minimizer, clipped to a step of at most `step` from the best. The step grows
    after an improvement and shrinks after a failure."""

    def __init__(self, space, seed=0, fit_radius=1.0, step=0.3, min_step=0.005):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.fit_radius = float(fit_radius)
        self.step = float(step)
        self.min_step = float(min_step)
        self.X, self.y = [], []
        self.pending = []
        self.last_best = None
        self.rng = np.random.default_rng(int(seed))

    def _features(self, P):
        P = np.atleast_2d(P)
        a, b = P[:, 0], P[:, 1]
        return np.c_[np.ones_like(a), a, b, a * a, a * b, b * b]

    def _propose(self):
        X = np.array(self.X)
        y = np.array(self.y)
        xb = X[int(np.argmin(y))]
        d = np.linalg.norm(X - xb, axis=1)
        idx = np.where(d <= self.fit_radius)[0]
        if len(idx) < 8:
            idx = np.argsort(d)[:8]
        w = np.exp(-(d[idx] / self.fit_radius) ** 2)
        M = self._features(X[idx]) * w[:, None]
        c = np.linalg.lstsq(M, y[idx] * w, rcond=None)[0]
        g = c[1:3]
        H = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
        ev, V = np.linalg.eigh(H)
        if ev.min() > 1e-8:
            x = np.linalg.solve(H, -g)
        else:
            gb = g + H @ xb
            x = xb - gb / (np.linalg.norm(gb) + 1e-12) * self.step
        s = x - xb
        n = np.linalg.norm(s)
        if n > self.step:
            x = xb + s / n * self.step
        x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        # deterministic objective: never repeat a point; nudge along the flat direction
        if np.min(np.linalg.norm(X - x, axis=1)) < 1e-3:
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            x = np.clip(xb + sign * V[:, 0] * max(self.step, self.min_step),
                        self.bounds[:, 0], self.bounds[:, 1])
        return list(map(float, x))

    def ask(self, n):
        out = []
        for _ in range(n):
            x = self._propose()
            self.pending.append(x)
            out.append(x)
        return out

    def tell(self, X, y, g):
        before = min(self.y) if self.y else np.inf
        asked = False
        for xi, yi in zip(X, y):
            xi = list(map(float, xi))
            for j, p in enumerate(self.pending):
                if np.allclose(p, xi):
                    self.pending.pop(j)
                    asked = True
                    break
            self.X.append(xi)
            self.y.append(float(yi))
        if asked:
            if min(self.y) < before:
                self.step = min(self.step * 1.5, 1.0)
            else:
                self.step = max(self.step * 0.5, self.min_step)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
