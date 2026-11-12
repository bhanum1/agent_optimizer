"""Trust-region optimizer driven by a locally weighted quadratic model.

Each ask fits a quadratic (separable terms free, cross terms ridge-penalized) to all
data, weighted by a Gaussian kernel around the incumbent, and proposes the model
minimum inside a box trust region. Every `explore_every`-th ask is instead a random
point in the trust region, to keep the local design well spread.
"""
import json
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, radius=1.0, rmin=1e-3, rmax=2.0, ridge=(1.0, 10.0, 100.0, 1e3, 1e4, 1e5),
                 explore_every=4, fail_shrink=3, min_eff=30.0, tol=1e-4):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lo)
        self.cfg = dict(radius0=radius, rmin=rmin, rmax=rmax, ridge=list(ridge),
                        explore_every=explore_every, fail_shrink=fail_shrink,
                        min_eff=min_eff, tol=tol)
        self.r = radius
        self.fails = 0
        self.nask = 0
        self.X, self.y = [], []
        self.pending = []
        self.rng = np.random.default_rng(seed)

    # ---- model ----
    def _feats(self, X):
        X = np.atleast_2d(X)
        iu = np.triu_indices(self.d, 1)
        cross = (X[:, :, None] * X[:, None, :])[:, iu[0], iu[1]]
        return np.c_[np.ones(len(X)), X, X ** 2, cross]

    def _fit(self, xb):
        X, y = np.array(self.X), np.array(self.y)
        dist = np.linalg.norm(X - xb, axis=1)
        L = max(2.5 * self.r, 0.3)
        while True:
            w = np.exp(-dist ** 2 / (2 * L * L))
            if w.sum() >= min(self.cfg["min_eff"], len(y)) or L > 50:
                break
            L *= 1.3
        c0 = xb  # center features at incumbent for conditioning
        A = self._feats(X - c0)
        p = A.shape[1]
        sw = np.sqrt(w)
        Aw, yw = A * sw[:, None], y * sw
        G = Aw.T @ Aw
        best = None
        # cross-term ridge chosen by weighted leave-one-out error
        for lam in self.cfg["ridge"]:
            pen = np.zeros(p)
            pen[1 + 2 * self.d:] = lam
            Minv = np.linalg.inv(G + np.diag(pen) + 1e-9 * np.eye(p))
            coef = Minv @ (Aw.T @ yw)
            h = np.einsum("ij,jk,ik->i", Aw, Minv, Aw)
            loo = (yw - Aw @ coef) / np.maximum(1 - h, 1e-6)
            err = np.sum(loo ** 2)
            if best is None or err < best[0]:
                best = (err, coef)
        coef = best[1]
        return lambda Z: self._feats(np.atleast_2d(Z) - c0) @ coef

    def _best(self):
        i = int(np.argmin(self.y))
        return np.array(self.X[i]), self.y[i]

    # ---- interface ----
    def ask(self, n):
        out = []
        for _ in range(n):
            out.append(self._ask1().tolist())
        self.pending += out
        return out

    def _ask1(self):
        self.nask += 1
        if len(self.y) < 2 * self.d + 2:
            return self.rng.uniform(self.lo, self.hi)
        xb, _ = self._best()
        lo = np.maximum(self.lo, xb - self.r)
        hi = np.minimum(self.hi, xb + self.r)
        if self.nask % self.cfg["explore_every"] == 0:
            return self.rng.uniform(lo, hi)
        f = self._fit(xb)
        starts = [xb] + [self.rng.uniform(lo, hi) for _ in range(8)]
        best = None
        for s in starts:
            res = minimize(lambda z: float(f(z)[0]), s, method="L-BFGS-B",
                           bounds=list(zip(lo, hi)))
            if best is None or res.fun < best.fun:
                best = res
        x = best.x
        X = np.array(self.X + self.pending) if self.pending else np.array(self.X)
        if np.min(np.linalg.norm(X - x, axis=1)) < 1e-3 * max(self.r, 1e-3) + 1e-6:
            x = np.clip(x + self.rng.normal(0, 0.1 * self.r, self.d), lo, hi)
        return x

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = list(map(float, x))
            v = float(v) if np.isfinite(v) else (max(self.y) + 1 if self.y else 1e10)
            prev = min(self.y) if self.y else np.inf
            self.X.append(x)
            self.y.append(v)
            if x in self.pending:
                self.pending.remove(x)
                if v < prev - self.cfg["tol"]:
                    self.fails = 0
                    self.r = min(self.cfg["rmax"], 1.5 * self.r)
                else:
                    self.fails += 1
                    if self.fails >= self.cfg["fail_shrink"]:
                        self.fails = 0
                        self.r *= 0.5
                        if self.r < self.cfg["rmin"]:
                            self.r = self.cfg["radius0"]

    def save(self, path):
        st = dict(lo=self.lo.tolist(), hi=self.hi.tolist(), cfg=self.cfg, r=self.r,
                  fails=self.fails, nask=self.nask, X=self.X, y=self.y,
                  pending=self.pending, rng=self.rng.bit_generator.state)
        with open(path, "w") as fh:
            json.dump(st, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            st = json.load(fh)
        o = cls.__new__(cls)
        o.lo, o.hi = np.array(st["lo"]), np.array(st["hi"])
        o.d = len(o.lo)
        o.cfg, o.r, o.fails, o.nask = st["cfg"], st["r"], st["fails"], st["nask"]
        o.X, o.y, o.pending = st["X"], st["y"], st["pending"]
        o.rng = np.random.default_rng()
        o.rng.bit_generator.state = st["rng"]
        return o
