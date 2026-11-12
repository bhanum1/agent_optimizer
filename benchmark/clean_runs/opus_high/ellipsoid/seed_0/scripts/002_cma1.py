"""CMA-ES (pycma) one point at a time, with restarts.

Warm start: the first run starts at the best told point. If `hess_init`, its
initial covariance is the inverse of the (eigen-clipped) Hessian of a full quadratic
fitted to the told data, normalized to geometric-mean eigenvalue 1, so sigma0 keeps
its meaning. On stop, restarts from the best point with sigma0 * restart_sigma_mult
and popsize * restart_pop_mult (IPOP-like), keeping the last run's covariance if
`keep_C`.
"""
import pickle
from itertools import combinations_with_replacement
import numpy as np
import cma


def _quad_hessian(X, y):
    n = X.shape[1]
    pairs = list(combinations_with_replacement(range(n), 2))
    F = np.column_stack([np.ones(len(X))] + [X[:, j] for j in range(n)]
                        + [X[:, a] * X[:, b] for a, b in pairs])
    c = np.linalg.lstsq(F, y, rcond=None)[0]
    H = np.zeros((n, n))
    for k, (a, b) in enumerate(pairs):
        v = c[1 + n + k]
        if a == b:
            H[a, a] = 2 * v
        else:
            H[a, b] = H[b, a] = v
    return H


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, sigma0=1.0, popsize=None, hess_init=True,
                 max_cond=1e8, restart_sigma_mult=1.0, restart_pop_mult=1.0,
                 keep_C=True, tol=1e-13):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.dim = space["dim"]
        self.seed = seed
        self.sigma0 = sigma0
        self.popsize = popsize or (4 + int(3 * np.log(self.dim)))
        self.hess_init = hess_init
        self.max_cond = max_cond
        self.rs_mult, self.rp_mult = restart_sigma_mult, restart_pop_mult
        self.keep_C = keep_C
        self.tol = tol
        self.X, self.y = [], []
        self.es = None
        self.nrun = 0
        self.pop, self.vals, self.next_i = None, None, 0

    def _C0(self):
        X, y = np.array(self.X), np.array(self.y)
        n = self.dim
        if not self.hess_init or len(y) < 2 * (n + 1) * (n + 2) // 2:
            return None
        H = _quad_hessian(X, y)
        w, V = np.linalg.eigh(H)
        if w.max() <= 0:
            return None
        w = np.maximum(w, w.max() / self.max_cond)
        C = (V / w) @ V.T
        return C / np.exp(np.mean(np.log(1 / w)))

    def _new_es(self):
        best = int(np.argmin(self.y)) if self.y else None
        x0 = np.array(self.X[best]) if best is not None else (self.lo + self.hi) / 2
        if self.es is None:
            C, sig, pop = self._C0(), self.sigma0, self.popsize
        else:
            C = self.es.sm.C / np.exp(np.mean(np.log(self.es.sm.eigenspectrum))) if self.keep_C else None
            sig = self.sigma0 * self.rs_mult ** self.nrun
            pop = int(round(self.popsize * self.rp_mult ** self.nrun))
        opts = {"bounds": [list(self.lo), list(self.hi)], "seed": self.seed + 1000 * self.nrun + 1,
                "verbose": -9, "popsize": pop, "tolfun": self.tol, "tolx": self.tol,
                "tolfunhist": self.tol, "maxfevals": 10 ** 7}
        self.es = cma.CMAEvolutionStrategy(list(x0), sig, opts)
        if C is not None:
            self.es.sm.C = C
            self.es.sm._decompose_C()
        self.nrun += 1
        self.pop, self.vals, self.next_i = None, None, 0

    def ask(self, n):
        if self.es is None:
            self._new_es()
        if self.pop is None:
            if self.es.stop():
                self._new_es()
            self.pop = self.es.ask()
            self.vals = [None] * len(self.pop)
            self.next_i = 0
        x = np.clip(self.pop[self.next_i], self.lo, self.hi)
        self.next_i += 1
        return [list(map(float, x))]

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append(list(map(float, xi)))
            self.y.append(float(yi))
            if self.pop is not None:
                for i in range(self.next_i):
                    if self.vals[i] is None and np.allclose(
                            np.clip(self.pop[i], self.lo, self.hi), xi, atol=1e-12, rtol=0):
                        self.vals[i] = float(yi)
                        break
        if self.pop is not None and all(v is not None for v in self.vals):
            self.es.tell(self.pop, self.vals)
            self.pop = None

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as fh:
            o.__dict__.update(pickle.load(fh))
        return o
