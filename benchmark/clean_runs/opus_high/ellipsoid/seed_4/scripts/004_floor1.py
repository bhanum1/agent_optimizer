"""Probe the floor of a stiff valley at many spread-out points.

The valley is {x : v.x ~= c} for a stiff unit direction v. For each base point
(Sobol in the box, projected onto the hyperplane v.x = c, kept if inside the box) it
measures t = 0, +delta, -delta along v, then the vertex of the parabola through them,
then the vertex of the parabola through the three points nearest that vertex. So each
probe costs 5 measurements and ends with a measured point on the valley floor.
"""
import pickle

import numpy as np
from scipy.stats import qmc


def _vertex(ts, ys, fallback_step):
    ts, ys = np.asarray(ts), np.asarray(ys)
    A = np.vstack([ts ** 2, ts, np.ones(len(ts))]).T
    a, b, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    if a > 0:
        t = -b / (2 * a)
        lo, hi = ts.min() - 10 * fallback_step, ts.max() + 10 * fallback_step
        return float(np.clip(t, lo, hi))
    i = int(np.argmin(ys))
    return float(ts[i] + (fallback_step * 3 if ts[i] >= np.median(ts) else -fallback_step * 3))


class Optimizer:
    max_ask = 1

    def __init__(self, space, v, c, n_probes=25, delta=0.1, seed=0, extra=None):
        self.lb = np.array([b[0] for b in space["bounds"]], float)
        self.ub = np.array([b[1] for b in space["bounds"]], float)
        v = np.asarray(v, float)
        self.v = v / np.linalg.norm(v)
        self.c = float(c)
        self.delta = delta
        sob = qmc.Sobol(len(self.lb), scramble=True, seed=seed)
        bases = [np.asarray(e, float) for e in (extra or [])]
        while len(bases) < n_probes:
            for p in sob.random(64):
                x = self.lb + p * (self.ub - self.lb)
                x = x + (self.c - self.v @ x) * self.v
                if np.all(x > self.lb + 0.2) and np.all(x < self.ub - 0.2):
                    bases.append(x)
                if len(bases) >= n_probes:
                    break
        self.bases = [b.tolist() for b in bases]
        self.k = -1
        self.ts, self.ys = [], []
        self.asked = None
        self.X, self.y = [], []
        self.results = []    # (base index, t, y) of each probe's final point

    def _pt(self, t):
        return np.clip(np.asarray(self.bases[self.k]) + t * self.v, self.lb, self.ub)

    def _next_t(self):
        while True:
            m = len(self.ts)
            if self.k >= 0 and m < 5:
                d = self.delta
                if m < 3:
                    return [0.0, d, -d][m]
                if m == 3:
                    return _vertex(self.ts, self.ys, d)
                near = np.argsort(np.abs(np.array(self.ts) - self.ts[-1]))[:3]
                t = _vertex(np.array(self.ts)[near], np.array(self.ys)[near], d / 4)
                if np.min(np.abs(np.array(self.ts) - t)) > 1e-6:
                    return t
            if self.k >= 0:
                i = int(np.argmin(self.ys))
                self.results.append((self.k, self.ts[i], self.ys[i]))
            if self.k + 1 >= len(self.bases):
                return None
            self.k += 1
            self.ts, self.ys = [], []

    def ask(self, n):
        t = self._next_t()
        if t is None:
            return []
        self.asked = t
        return [self._pt(t).tolist()]

    def tell(self, X, y, g=None):
        for xi, yi in zip(X, y):
            self.X.append(list(map(float, xi)))
            self.y.append(float(yi))
            if self.asked is not None and np.allclose(self._pt(self.asked), xi, atol=1e-12, rtol=0):
                self.ts.append(self.asked)
                self.ys.append(float(yi))
                self.asked = None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
