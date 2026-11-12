"""Quadratic-model trust-region search, one point at a time.

Each ask fits a full quadratic (least squares) to the `k` told points nearest the
incumbent and proposes the model's minimizer inside a Euclidean ball of radius
`delta` around the incumbent, clipped to the box. Every `geo_every`-th ask (and when
the model step is degenerate) it instead proposes a geometry point: incumbent plus
delta along the eigenvector of the model Hessian least represented in the local
sample. delta doubles on a good model step and halves on a failed one.
"""
import pickle
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import minimize


def _features(D):
    n = D.shape[1]
    pairs = list(combinations_with_replacement(range(n), 2))
    return np.column_stack([np.ones(len(D))] + [D[:, j] for j in range(n)]
                           + [D[:, a] * D[:, b] for a, b in pairs]), pairs


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, delta0=0.05, k=60, geo_every=4,
                 delta_min=1e-7, delta_max=1.0):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.n = space["dim"]
        self.rng = np.random.default_rng(seed)
        self.delta = delta0
        self.delta0 = delta0
        self.k, self.geo_every = k, geo_every
        self.delta_min, self.delta_max = delta_min, delta_max
        self.X, self.y = [], []
        self.it = 0
        self.pending = None  # (x, kind, predicted reduction)

    def _fit(self, c):
        X, y = np.array(self.X), np.array(self.y)
        d = np.linalg.norm(X - c, axis=1)
        idx = np.argsort(d)[: self.k]
        D = X[idx] - c
        F, pairs = _features(D)
        coef = np.linalg.lstsq(F, y[idx], rcond=None)[0]
        g = coef[1: 1 + self.n]
        H = np.zeros((self.n, self.n))
        for j, (a, bb) in enumerate(pairs):
            v = coef[1 + self.n + j]
            if a == bb:
                H[a, a] = 2 * v
            else:
                H[a, bb] = H[bb, a] = v
        return g, H, D

    def _tr_step(self, c, g, H):
        lo, hi, r = self.lo - c, self.hi - c, self.delta
        m = lambda s: g @ s + 0.5 * s @ H @ s
        jac = lambda s: g + H @ s
        cons = [{"type": "ineq", "fun": lambda s: r * r - s @ s, "jac": lambda s: -2 * s}]
        starts = [np.zeros(self.n), np.clip(-g / (np.linalg.norm(g) + 1e-300) * r, lo, hi)]
        try:
            sn = np.linalg.solve(H, -g)
            nn = np.linalg.norm(sn)
            starts.append(np.clip(sn * min(1, r / nn), lo, hi))
        except np.linalg.LinAlgError:
            pass
        best = None
        for s0 in starts:
            res = minimize(m, s0, jac=jac, method="SLSQP", constraints=cons,
                           bounds=list(zip(lo, hi)), options={"maxiter": 200, "ftol": 1e-15})
            s = np.clip(res.x, lo, hi)
            nr = np.linalg.norm(s)
            if nr > r:
                s = s * r / nr
            if best is None or m(s) < m(best):
                best = s
        return best, -m(best)

    def _geo_point(self, c, H, D):
        w, V = np.linalg.eigh(H)
        # spread of local sample along each eigenvector, relative to delta
        spread = np.sqrt(np.mean((D @ V) ** 2, axis=0)) + 1e-300
        j = int(np.argmin(spread * np.sqrt(np.abs(w) + 1e-12)))
        sgn = self.rng.choice([-1.0, 1.0])
        x = np.clip(c + sgn * self.delta * V[:, j], self.lo, self.hi)
        return x

    def ask(self, n):
        if self.pending is not None:
            return [list(self.pending[0])]
        ib = int(np.argmin(self.y))
        c = np.array(self.X[ib])
        g, H, D = self._fit(c)
        self.it += 1
        s, pred = self._tr_step(c, g, H)
        X = np.array(self.X)
        mind = np.min(np.linalg.norm(X - (c + s), axis=1))
        if (self.geo_every and self.it % self.geo_every == 0) or pred <= 0 or mind < 1e-3 * self.delta:
            x, kind = self._geo_point(c, H, D), "geo"
            pred = 0.0
        else:
            x, kind = c + s, "tr"
        self.pending = (x, kind, pred, float(self.y[ib]), float(np.linalg.norm(s)))
        return [list(x)]

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append(list(map(float, xi)))
            self.y.append(float(yi))
        if self.pending is None or len(X) != 1:
            return
        x, kind, pred, fbest, snorm = self.pending
        self.pending = None
        if kind != "tr":
            return
        actual = fbest - float(y[0])
        rho = actual / pred if pred > 0 else -1
        if rho > 0.7 and snorm > 0.9 * self.delta:
            self.delta = min(2 * self.delta, self.delta_max)
        elif rho < 0.1:
            self.delta *= 0.5
        if self.delta < self.delta_min:
            self.delta = self.delta0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
