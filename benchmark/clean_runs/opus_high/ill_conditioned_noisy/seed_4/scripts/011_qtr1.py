# Sequential quadratic trust-region search around the best point, fitted on all told data.
# Each ask fits a weighted quadratic to the points nearest the best and steps to the minimizer
# of the model within radius r. Success expands r, failure halves it and queues geometry points
# along the model's eigen-directions, sized to a predicted rise of `delta` (at most rmax).
import pickle
import numpy as np

IU = np.triu_indices(5)


class Optimizer:
    max_ask = 1

    def __init__(self, space, r0=0.1, k=40, delta=1.0, rmin=0.005, rmax=0.5, ycap=None, gauss=True, seed=0):
        b = np.array(space["bounds"], dtype=float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.d = len(b)
        self.r, self.k, self.delta = r0, k, delta
        self.rmin, self.rmax = rmin, rmax
        self.ycap, self.gauss = ycap, gauss   # fit only points with y < best + ycap; Gaussian or uniform weights
        self.rng = np.random.default_rng(seed)
        self.X, self.y = [], []
        self.queue = []          # geometry points to ask next
        self.pending = None      # (x, kind, best_y_before, step_norm)
        self.gi = 0              # geometry direction counter

    def _features(self, Z):
        iu = np.triu_indices(self.d)
        return np.hstack([np.ones((len(Z), 1)), Z, np.array([np.outer(z, z)[iu] for z in Z])])

    def _fit(self):
        X, y = np.array(self.X), np.array(self.y)
        c = X[y.argmin()]
        dist = np.linalg.norm(X - c, axis=1)
        if self.ycap is not None:
            dist = np.where(y < y.min() + self.ycap, dist, np.inf)
        idx = np.argsort(dist)[: self.k]
        h = max(dist[idx[-1]], 1e-6)
        w = np.exp(-(dist[idx] / h) ** 2) if self.gauss else np.ones(len(idx))
        Z = X[idx] - c
        F = self._features(Z)
        sw = np.sqrt(w)[:, None]
        coef = np.linalg.lstsq(F * sw, y[idx] * sw[:, 0], rcond=None)[0]
        iu = np.triu_indices(self.d)
        H = np.zeros((self.d, self.d))
        H[iu] = coef[1 + self.d:]
        H = H + H.T
        return c, y.min(), coef[0], coef[1:1 + self.d], H

    def _trstep(self, g, H, rad):
        ev, V = np.linalg.eigh(H)
        gt = V.T @ g
        lo = max(0.0, -ev.min()) + 1e-12
        s = lambda lam: -gt / (ev + lam)
        if ev.min() > 0 and np.linalg.norm(s(0.0)) <= rad:
            return V @ s(0.0)
        hi = lo + 1.0
        while np.linalg.norm(s(hi)) > rad:
            hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if np.linalg.norm(s(mid)) > rad:
                lo = mid
            else:
                hi = mid
        return V @ s(hi)

    def _geometry(self, c, H, n):
        ev, V = np.linalg.eigh(H)
        out = []
        for _ in range(n):
            i = (self.gi // 2) % self.d
            sign = 1 if self.gi % 2 == 0 else -1
            self.gi += 1
            step = min(self.rmax, np.sqrt(2 * self.delta / max(ev[i], 1e-9)))
            out.append(np.clip(c + sign * step * V[:, i], self.lo, self.hi))
        return out

    def ask(self, n):
        if self.queue:
            x = self.queue.pop(0)
            kind, sn = "geo", 0.0
        else:
            c, ybest, f0, g, H = self._fit()
            s = self._trstep(g, H, self.r)
            x = np.clip(c + s, self.lo, self.hi)
            if np.min(np.linalg.norm(np.array(self.X) - x, axis=1)) < 1e-6:
                x = np.clip(c + self.rng.normal(size=self.d) * self.r * 0.1, self.lo, self.hi)
            kind, sn = "tr", float(np.linalg.norm(x - c))
        self.pending = (np.array(x, dtype=float), kind, float(min(self.y)), sn)
        return [list(map(float, x))]

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            xi = np.array(xi, dtype=float)
            self.X.append(xi)
            self.y.append(float(yi))
            if self.pending is not None and np.allclose(xi, self.pending[0]):
                _, kind, yb, sn = self.pending
                self.pending = None
                if kind == "tr":
                    if yi < yb - 1e-3:
                        self.r = min(self.rmax, max(self.r, 2 * sn))
                    else:
                        self.r = max(self.rmin, 0.5 * min(self.r, max(sn, self.rmin)))
                        c, _, _, _, H = self._fit()
                        self.queue += self._geometry(c, H, 2)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as fh:
            o.__dict__.update(pickle.load(fh))
        return o
