"""Trust-region method with a full quadratic model fitted by least squares.

Each ask fits a full quadratic (21 terms in 5-D) to the measured points nearest the
incumbent, solves the trust-region subproblem exactly (eigendecomposition), and
proposes that step. The radius grows on good agreement and shrinks on poor. When
there are too few nearby points, or the fit is ill-conditioned, it proposes a
geometry point at the radius instead. All data told (warm start included) feeds the
model, so the method uses every earlier measurement near the incumbent.
"""
import pickle
from itertools import combinations_with_replacement as cwr

import numpy as np


def _feats(Z):
    n, d = Z.shape
    cols = [np.ones(n)] + [Z[:, i] for i in range(d)] + [Z[:, i] * Z[:, j] for i, j in cwr(range(d), 2)]
    return np.array(cols).T


def _unpack(c, d):
    g = c[1:d + 1]
    H = np.zeros((d, d))
    k = d + 1
    for i, j in cwr(range(d), 2):
        if i == j:
            H[i, i] = 2 * c[k]
        else:
            H[i, j] = H[j, i] = c[k]
        k += 1
    return g, H


def _tr_step(g, H, delta):
    """Minimize g.s + s.H.s/2 subject to |s| <= delta."""
    ev, V = np.linalg.eigh(H)
    gt = V.T @ g
    if ev[0] > 0:
        s = -gt / ev
        if np.linalg.norm(s) <= delta:
            return V @ s
    lo = max(0.0, -ev[0]) + 1e-12
    hi = lo + np.linalg.norm(g) / delta + abs(ev).max() + 1.0

    def nrm(lam):
        return np.linalg.norm(gt / (ev + lam))
    if nrm(lo) < delta:  # hard case: go along the most negative eigenvector
        s = -gt / (ev + lo)
        s[0] += np.sqrt(max(delta ** 2 - np.dot(s, s), 0.0)) * (1 if gt[0] <= 0 else -1)
        return V @ s
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if nrm(mid) > delta:
            lo = mid
        else:
            hi = mid
    return V @ (-gt / (ev + hi))


class Optimizer:
    max_ask = 1

    def __init__(self, space, delta0=0.01, delta_min=1e-7, delta_max=1.0, nfit=40,
                 fit_mult=4.0, seed=0):
        self.lb = np.array([b[0] for b in space["bounds"]], float)
        self.ub = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lb)
        self.delta = delta0
        self.delta0 = delta0
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.nfit = nfit
        self.fit_mult = fit_mult
        self.rng = np.random.default_rng(seed)
        self.X, self.y = [], []
        self.pending = None   # (x, kind, predicted reduction, incumbent y)
        self.nfail = 0

    def _fit(self, b):
        X = np.array(self.X)
        y = np.array(self.y)
        dist = np.linalg.norm(X - b, axis=1)
        order = np.argsort(dist)
        p = (self.d + 1) * (self.d + 2) // 2
        rad = max(self.fit_mult * self.delta, dist[order[min(len(order), p + 5) - 1]])
        idx = order[dist[order] <= rad][: max(self.nfit, p + 5)]
        if len(idx) < p + 3:
            return None
        sc = max(np.max(dist[idx]), 1e-12)
        Z = (X[idx] - b) / sc
        F = _feats(Z)
        sv = np.linalg.svd(F, compute_uv=False)
        if sv[-1] / sv[0] < 1e-8:
            return None
        c, *_ = np.linalg.lstsq(F, y[idx] - y[idx].min(), rcond=None)
        g, H = _unpack(c, self.d)
        return g / sc, H / sc ** 2

    def ask(self, n):
        y = np.array(self.y) if self.y else None
        if y is None:
            x = (self.lb + self.ub) / 2
            self.pending = (x, "geo", 0.0, np.inf)
            return [x.tolist()]
        ib = int(np.argmin(y))
        b = np.array(self.X[ib])
        model = self._fit(b)
        x = None
        if model is not None:
            g, H = model
            s = _tr_step(g, H, self.delta)
            xs = np.clip(b + s, self.lb, self.ub)
            s = xs - b
            pred = -(g @ s + 0.5 * s @ H @ s)
            if pred > 0 and np.linalg.norm(s) > 1e-14 and not self._seen(xs):
                x = xs
                self.pending = (x, "tr", float(pred), float(y[ib]), float(np.linalg.norm(s)))
        if x is None:
            u = self.rng.standard_normal(self.d)
            u /= np.linalg.norm(u)
            x = np.clip(b + self.delta * u, self.lb, self.ub)
            self.pending = (x, "geo", 0.0, float(y[ib]), self.delta)
        return [x.tolist()]

    def _seen(self, x):
        if not self.X:
            return False
        return np.min(np.linalg.norm(np.array(self.X) - x, axis=1)) < 1e-12

    def tell(self, X, y, g=None):
        for xi, yi in zip(X, y):
            self.X.append(list(map(float, xi)))
            self.y.append(float(yi))
            p = self.pending
            if p is not None and np.allclose(p[0], xi, atol=1e-12, rtol=0):
                self.pending = None
                if p[1] == "tr":
                    rho = (p[3] - yi) / p[2]
                    if rho > 0.75 and p[4] > 0.9 * self.delta:
                        self.delta = min(2 * self.delta, self.delta_max)
                    elif rho < 0.1:
                        self.delta = max(0.5 * self.delta, self.delta_min)
                    if self.delta <= self.delta_min * 1.0001:
                        self.delta = self.delta0  # reset after collapse
        # nothing else

    def save(self, path):
        st = dict(self.__dict__)
        st["rng"] = self.rng.bit_generator.state
        with open(path, "wb") as f:
            pickle.dump(st, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            st = pickle.load(f)
        rs = st.pop("rng")
        obj.__dict__.update(st)
        obj.rng = np.random.default_rng()
        obj.rng.bit_generator.state = rs
        return obj
