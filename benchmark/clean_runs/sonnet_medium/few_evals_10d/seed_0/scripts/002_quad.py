import pickle
import numpy as np
from scipy.optimize import least_squares


class Optimizer:
    def __init__(self, space, seed=0, sigma0=0.6, sigma1=0.15, horizon=40):
        self.b = np.array(space["bounds"], float)
        self.d = len(self.b)
        self.rng = np.random.default_rng(seed)
        self.s0, self.s1, self.h = sigma0, sigma1, horizon
        self.X, self.y, self.pending, self.n = [], [], [], 0
        self.c = None

    def _fit(self):
        X, y = np.array(self.X), np.array(self.y)
        d = self.d
        if len(y) < 2 * d + 2:
            return
        lo, hi = self.b[:, 0], self.b[:, 1]
        c0 = np.clip(self.c if self.c is not None else (lo + hi) / 2, lo, hi)
        p0 = np.r_[np.ones(d), c0, y.min() - 50]
        r = lambda p: (p[:d] * (X - p[d:2 * d]) ** 2).sum(1) + p[-1] - y
        lb = np.r_[np.full(d, 0.01), lo - 5, -np.inf]
        ub = np.r_[np.full(d, 100.0), hi + 5, np.inf]
        try:
            s = least_squares(r, p0, bounds=(lb, ub))
            self.c = np.clip(s.x[d:2 * d], lo, hi)
        except Exception:
            pass

    def ask(self, n):
        lo, hi = self.b[:, 0], self.b[:, 1]
        out = []
        for _ in range(n):
            frac = min(self.n / self.h, 1.0)
            sig = self.s0 * (self.s1 / self.s0) ** frac
            if self.c is None:
                x = self.rng.uniform(lo, hi)
            else:
                x = self.c + sig * self.rng.standard_normal(self.d)
            x = np.clip(x, lo, hi)
            self.n += 1
            self.pending.append(x.tolist())
            out.append(x.tolist())
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            self.X.append(list(x))
            self.y.append(float(v))
        self._fit()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
