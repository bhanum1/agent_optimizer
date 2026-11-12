"""Local refinement by a weighted full-quadratic model around the incumbent.

Each ask fits y ~ quadratic(x) on the n_fit points nearest the centre (Gaussian
weights of width `width`, small ridge), takes the model minimizer, clips the step
to the trust radius `rho`, and adds jitter of size rho*jitter. The centre is the
measured point with the lowest model value. rho shrinks after `fail_tol` asks
without a new best measurement, and grows after a success.
"""
import json, os, tempfile
from itertools import combinations_with_replacement
import numpy as np


class Optimizer:
    def __init__(self, space, seed=0, rho=0.05, rho_min=0.002, rho_max=0.3,
                 n_fit=120, width=1.0, ridge=1e-6, jitter=0.3, fail_tol=4):
        self.lo = [b[0] for b in space["bounds"]]
        self.hi = [b[1] for b in space["bounds"]]
        self.cfg = dict(rho=rho, rho_min=rho_min, rho_max=rho_max, n_fit=n_fit,
                        width=width, ridge=ridge, jitter=jitter, fail_tol=fail_tol)
        self.rho = rho
        self.fails = 0
        self.X, self.y = [], []
        self.best = None
        self.pending = []
        self.rng_state = np.random.default_rng(seed).bit_generator.state

    # quadratic features of z = x - centre
    @staticmethod
    def _feats(Z):
        d = Z.shape[1]
        cols = [np.ones(len(Z))] + [Z[:, i] for i in range(d)]
        cols += [Z[:, i] * Z[:, j] for i, j in combinations_with_replacement(range(d), 2)]
        return np.stack(cols, 1)

    def _fit(self, c):
        X, y = np.asarray(self.X), np.asarray(self.y)
        d = X.shape[1]
        dist = np.linalg.norm(X - c, axis=1)
        idx = np.argsort(dist)[: self.cfg["n_fit"]]
        Z = X[idx] - c
        w = np.exp(-0.5 * (dist[idx] / self.cfg["width"]) ** 2)
        F = self._feats(Z)
        sw = np.sqrt(w)[:, None]
        A = F * sw
        reg = self.cfg["ridge"] * np.eye(F.shape[1])
        reg[0, 0] = 0
        coef = np.linalg.solve(A.T @ A + reg, A.T @ (y[idx] * sw[:, 0]))
        g = coef[1 : d + 1]
        H = np.zeros((d, d))
        k = d + 1
        for i, j in combinations_with_replacement(range(d), 2):
            if i == j:
                H[i, i] = 2 * coef[k]
            else:
                H[i, j] = H[j, i] = coef[k]
            k += 1
        return coef, g, H

    def ask(self, n):
        rng = np.random.default_rng()
        rng.bit_generator.state = self.rng_state
        out = []
        X, y = np.asarray(self.X), np.asarray(self.y)
        d = len(self.lo)
        for _ in range(n):
            ok = len(self.y) >= 70
            if ok:
                try:
                    c0 = X[np.argmin(y)]
                    coef, g, H = self._fit(c0)
                    # centre: lowest model value among nearby measured points
                    near = np.linalg.norm(X - c0, axis=1) < 0.3
                    Zn = X[near] - c0
                    mv = self._feats(Zn) @ coef
                    c = X[near][np.argmin(mv)]
                    coef, g, H = self._fit(c)
                    ev, V = np.linalg.eigh(H)
                    ev = np.maximum(ev, 1e-3 * max(ev.max(), 1e-3))
                    s = -V @ ((V.T @ g) / ev)
                    ns = np.linalg.norm(s)
                    if ns > self.rho:
                        s *= self.rho / ns
                    s = s + rng.normal(size=d) * self.rho * self.cfg["jitter"] / np.sqrt(d)
                    x = c + s
                except Exception:
                    ok = False
            if not ok:
                c = X[np.argmin(y)] if len(y) else (np.array(self.lo) + np.array(self.hi)) / 2
                x = c + rng.normal(size=d) * self.rho / np.sqrt(d)
            x = np.clip(x, self.lo, self.hi)
            out.append([float(v) for v in x])
        self.pending.extend(out)
        self.rng_state = rng.bit_generator.state
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v)
            improved = self.best is not None and v < self.best
            self.X.append([float(t) for t in x])
            self.y.append(v)
            if self.best is None or v < self.best:
                self.best = v
            if x in self.pending:
                self.pending.remove(x)
                if improved:
                    self.fails = 0
                    self.rho = min(self.rho * 1.5, self.cfg["rho_max"])
                else:
                    self.fails += 1
                    if self.fails >= self.cfg["fail_tol"]:
                        self.fails = 0
                        self.rho = max(self.rho / 2, self.cfg["rho_min"])

    def save(self, path):
        st = dict(lo=self.lo, hi=self.hi, cfg=self.cfg, rho=self.rho, fails=self.fails,
                  X=self.X, y=self.y, best=self.best, pending=self.pending,
                  rng_state=self.rng_state)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)))
        with os.fdopen(fd, "w") as f:
            json.dump(st, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            st = json.load(f)
        o = cls.__new__(cls)
        for k, v in st.items():
            setattr(o, k, v)
        return o
