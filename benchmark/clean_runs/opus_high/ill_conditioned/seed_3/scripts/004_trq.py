import pickle
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Trust-region method with a local quadratic model fitted by weighted least
    squares to the measured points nearest the incumbent. Steps to the model's
    minimum within the trust radius (and the box); radius grows on good steps,
    shrinks on bad ones, and resets to delta0 when it collapses. Every few
    failed steps a geometry point at a random direction is added instead."""
    max_ask = 1
    batch = False

    def __init__(self, space, delta0=0.1, delta_min=1e-7, seed=0, kmult=4):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lo)
        self.p = (self.d + 1) * (self.d + 2) // 2
        self.delta0 = float(delta0)
        self.delta = float(delta0)
        self.delta_min = float(delta_min)
        self.kmult = int(kmult)
        self.rng = np.random.default_rng(seed)
        self.X, self.y = [], []
        self.pending = None   # (x, predicted decrease, kind)
        self.fails = 0

    def _feats(self, Z):
        cols = [np.ones(len(Z))] + [Z[:, i] for i in range(self.d)]
        for i in range(self.d):
            for j in range(i, self.d):
                cols.append(Z[:, i] * Z[:, j])
        return np.array(cols).T

    def _model(self, xc):
        X = np.array(self.X); y = np.array(self.y)
        dist = np.linalg.norm(X - xc, axis=1)
        K = min(len(X), self.kmult * self.p)
        idx = np.argsort(dist)[:K]
        s = max(self.delta, 1e-12)
        Z = (X[idx] - xc) / s
        w = 1.0 / (1.0 + (dist[idx] / (2 * s)) ** 2)
        F = self._feats(Z)
        c = np.linalg.lstsq(F * w[:, None], (y[idx] - y[idx].min()) * w, rcond=None)[0]
        g = c[1:1 + self.d]; H = np.zeros((self.d, self.d)); k = 1 + self.d
        for i in range(self.d):
            for j in range(i, self.d):
                if i == j:
                    H[i, i] = 2 * c[k]
                else:
                    H[i, j] = H[j, i] = c[k]
                k += 1
        return g, H, s

    def _step(self, xc):
        g, H, s = self._model(xc)
        lo = (self.lo - xc) / s; hi = (self.hi - xc) / s
        f = lambda u: g @ u + 0.5 * u @ H @ u
        jac = lambda u: g + H @ u
        cons = [{"type": "ineq", "fun": lambda u: 1.0 - u @ u, "jac": lambda u: -2 * u}]
        starts = [np.zeros(self.d)]
        gn = np.linalg.norm(g)
        if gn > 0:
            starts.append(-g / gn)
        ev, V = np.linalg.eigh(H)
        starts += [V[:, 0], -V[:, 0]]
        best = None
        for u0 in starts:
            u0 = np.clip(u0, lo, hi)
            try:
                r = minimize(f, u0, jac=jac, bounds=list(zip(lo, hi)), constraints=cons,
                             method="SLSQP", options={"maxiter": 200})
                u = np.clip(r.x, lo, hi)
                if u @ u > 1.0 + 1e-6:
                    u = u / np.linalg.norm(u)
                    u = np.clip(u, lo, hi)
                if best is None or f(u) < best[1]:
                    best = (u, f(u))
            except Exception:
                pass
        u, fu = best
        return np.clip(xc + s * u, self.lo, self.hi), -fu

    def ask(self, n):
        if self.pending is None:
            if not self.X:
                x = (self.lo + self.hi) / 2
                self.pending = (list(map(float, x)), 0.0, "init")
            else:
                ib = int(np.argmin(self.y)); xc = np.array(self.X[ib])
                kind = "step"
                if len(self.X) < self.p + 1 or self.fails >= 3:
                    kind = "geom"
                if kind == "step":
                    x, pred = self._step(xc)
                    near = np.min(np.linalg.norm(np.array(self.X) - x, axis=1))
                    if pred <= 0 or near < 1e-3 * self.delta:
                        kind = "geom"
                if kind == "geom":
                    v = self.rng.normal(size=self.d); v /= np.linalg.norm(v)
                    x = np.clip(xc + self.delta * v, self.lo, self.hi); pred = 0.0
                    self.fails = 0
                self.pending = (list(map(float, x)), float(pred), kind)
        return [self.pending[0]]

    def tell(self, X, y, g):
        fbest = min(self.y) if self.y else np.inf
        for x, v in zip(X, y):
            v = float(v)
            if not np.isfinite(v):
                v = 1e300
            self.X.append(list(map(float, x))); self.y.append(v)
        if self.pending is not None and len(X) == 1 and np.allclose(X[0], self.pending[0]):
            _, pred, kind = self.pending
            v = float(y[0])
            if kind == "step":
                rho = (fbest - v) / pred if pred > 0 else -1
                if rho > 0.7:
                    self.delta *= 2.0; self.fails = 0
                elif rho > 0.1:
                    self.fails = 0
                else:
                    self.delta *= 0.5; self.fails += 1
            elif kind == "geom" and v < fbest:
                self.delta *= 1.5
            if self.delta < self.delta_min:
                self.delta = self.delta0
            self.delta = min(self.delta, 0.5 * float(np.min(self.hi - self.lo)))
        self.pending = None if (self.pending is None or len(X) != 1 or np.allclose(X[0], self.pending[0])) else self.pending

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
