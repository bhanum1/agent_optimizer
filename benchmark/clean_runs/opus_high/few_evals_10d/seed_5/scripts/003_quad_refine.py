"""Local quadratic-model trust-region refinement.

Each ask fits a quadratic (linear + diagonal + ridge-shrunk cross terms) by least
squares to the `k` observed points nearest the current best, and proposes the
model's minimizer, clipped to a box of half-width `radius` around the best point.
Every `geo_every`-th ask is instead a geometry point: the best point plus a random
perturbation of size `radius`, so the model stays well conditioned. The radius
grows after an improvement and shrinks after `fail_tol` asks without one.
"""
import pickle

import numpy as np


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, k=60, radius=0.3, r_min=0.005, r_max=1.0,
                 geo_every=3, fail_tol=3, ridge=1.0):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.dim = len(b)
        self.rng = np.random.default_rng(seed)
        self.k, self.radius, self.r_min, self.r_max = k, radius, r_min, r_max
        self.geo_every, self.fail_tol, self.ridge = geo_every, fail_tol, ridge
        self.X, self.y = [], []
        self.n_ask = 0
        self.fail = 0
        self.history = []

    def _features(self, D):
        n, d = D.shape
        iu = np.triu_indices(d, 1)
        cross = (D[:, :, None] * D[:, None, :])[:, iu[0], iu[1]]
        return np.hstack([np.ones((n, 1)), D, D ** 2, cross]), iu

    def _vertex(self, c):
        X, y = np.array(self.X), np.array(self.y)
        dist = np.linalg.norm(X - c, axis=1)
        idx = np.argsort(dist)[:self.k]
        D, yy = X[idx] - c, y[idx]
        scale = max(np.abs(D).max(), 1e-6)
        F, iu = self._features(D / scale)
        d = self.dim
        pen = np.zeros(F.shape[1])
        pen[1 + 2 * d:] = self.ridge
        coef = np.linalg.solve(F.T @ F + np.diag(pen) + 1e-10 * np.eye(F.shape[1]), F.T @ yy)
        g = coef[1:1 + d]
        H = np.diag(2 * coef[1 + d:1 + 2 * d])
        H[iu] = coef[1 + 2 * d:]
        H[(iu[1], iu[0])] = coef[1 + 2 * d:]
        w, V = np.linalg.eigh(H)
        w = np.maximum(w, 1e-3 * max(w.max(), 1e-6))
        step = -V @ ((V.T @ g) / w)
        return step * scale

    def ask(self, n):
        out = []
        for _ in range(n):
            self.n_ask += 1
            X, y = np.array(self.X), np.array(self.y)
            c = X[int(np.argmin(y))]
            if self.geo_every and self.n_ask % self.geo_every == 0:
                step = self.radius * self.rng.uniform(-1, 1, self.dim)
            else:
                step = np.clip(self._vertex(c), -self.radius, self.radius)
            x = np.clip(c + step, self.lo, self.hi)
            out.append([float(v) for v in x])
        return out

    def tell(self, X, y, g):
        if not len(y):
            return
        own = len(y) == 1 and self.X
        best = min(self.y) if self.y else np.inf
        for x, v in zip(X, y):
            self.X.append([float(a) for a in x])
            self.y.append(float(v))
        if own:
            if float(y[0]) < best:
                self.radius = min(self.radius * 1.5, self.r_max)
                self.fail = 0
            else:
                self.fail += 1
                if self.fail >= self.fail_tol:
                    self.radius = max(self.radius / 2, self.r_min)
                    self.fail = 0
            self.history.append(self.radius)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
