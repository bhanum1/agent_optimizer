"""Quadratic-regression trust region: fit a ridge quadratic on the K points nearest the
incumbent, minimize it inside a ball around the incumbent, measure, adapt the radius.
Regression (not interpolation) smooths over small plateau steps."""
import pickle
from itertools import combinations_with_replacement as cwr
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, space, seed=0, radius=0.3, r_min=0.01, r_max=1.0, k=80,
                 fail_tol=3, ridge=1e-3, explore=0.15):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.d = len(self.lo)
        self.r0, self.r, self.r_min, self.r_max = radius, radius, r_min, r_max
        self.k, self.fail_tol, self.ridge, self.explore = k, fail_tol, ridge, explore
        self.fails = 0
        self.X, self.y = [], []
        self.pending = []
        self.rng = np.random.default_rng(seed)

    def _feats(self, Z):
        F = [np.ones(len(Z))] + [Z[:, i] for i in range(self.d)]
        F += [Z[:, i] * Z[:, j] for i, j in cwr(range(self.d), 2)]
        return np.array(F).T

    def _propose(self):
        X, y = np.array(self.X), np.array(self.y)
        c = X[np.argmin(y)]
        if len(y) < 2 * self._feats(c[None]).shape[1] or self.rng.random() < self.explore:
            u = self.rng.normal(size=self.d)
            return np.clip(c + self.r * self.rng.random() ** (1 / self.d) * u / np.linalg.norm(u),
                           self.lo, self.hi)
        dist = np.linalg.norm(X - c, axis=1)
        idx = np.argsort(dist)[: self.k]
        s = max(dist[idx].max(), 1e-9)
        Z = (X[idx] - c) / s
        w = np.exp(-(dist[idx] / s) ** 2)
        F = self._feats(Z)
        A = F.T @ (F * w[:, None]) + self.ridge * np.eye(F.shape[1])
        coef = np.linalg.solve(A, F.T @ (w * y[idx]))
        m = lambda z: self._feats(z[None])[0] @ coef
        rr = self.r / s
        cons = {"type": "ineq", "fun": lambda z: rr ** 2 - z @ z}
        best, bz = np.inf, np.zeros(self.d)
        for t in range(6):
            z0 = self.rng.normal(size=self.d) * rr / 2 if t else np.zeros(self.d)
            res = minimize(m, z0, constraints=[cons], method="SLSQP")
            if res.fun < best and res.x @ res.x <= rr ** 2 * 1.01:
                best, bz = res.fun, res.x
        x = np.clip(c + bz * s, self.lo, self.hi)
        if np.min(np.linalg.norm(X - x, axis=1)) < 1e-3 * max(self.r, 1e-3):
            u = self.rng.normal(size=self.d)
            x = np.clip(c + 0.5 * self.r * u / np.linalg.norm(u), self.lo, self.hi)
        return x

    def ask(self, n):
        out = []
        for _ in range(n):
            x = self._propose()
            self.pending.append(x.tolist())
            out.append(x.tolist())
        return out

    def tell(self, X, y, g):
        best = min(self.y) if self.y else np.inf
        for x, v in zip(X, y):
            v = float(v)
            if not np.isfinite(v):
                v = (max(self.y) if self.y else 0.0) + 1.0
            if x in self.pending:
                self.pending.remove(x)
            self.X.append(list(map(float, x)))
            self.y.append(v)
        if len(X) > 1:  # warm start batch: no radius update
            return
        if min(self.y) < best - 1e-9:
            self.fails = 0
            self.r = min(self.r * 1.5, self.r_max)
        else:
            self.fails += 1
            if self.fails >= self.fail_tol:
                self.fails = 0
                self.r *= 0.5
                if self.r < self.r_min:
                    self.r = self.r0
                    self.rng.normal(size=7)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
