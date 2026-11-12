import pickle
import numpy as np
import cma


class Optimizer:
    """Local CMA-ES around x0 with a given initial covariance C0, run in the
    whitened coordinates x = x0 + L u (L L^T = C0). Points outside the box are
    clipped; CMA is told the clipped value plus a quadratic penalty. When a run
    stops it restarts from the best point seen with the same L and sigma0."""

    def __init__(self, space, x0, C0, sigma0=0.05, seed=0, popsize=None,
                 penalty=1e6):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.dim = len(self.lo)
        self.x0 = np.array(x0, float)
        ev, V = np.linalg.eigh(np.array(C0, float))
        self.L = V @ np.diag(np.sqrt(np.maximum(ev, 1e-14)))
        self.sigma0 = sigma0
        self.rng = np.random.default_rng(seed)
        self.popsize = popsize or (4 + int(3 * np.log(self.dim)))
        self.penalty = penalty
        self.best_x, self.best_y = None, np.inf
        self.es = None
        self.center = self.x0.copy()
        self.queue, self.pending, self.pop, self.vals, self.xs = [], {}, [], [], []
        self.n_restarts = 0

    def _new_es(self, center):
        self.center = np.array(center, float)
        opts = {"popsize": self.popsize, "verbose": -9,
                "seed": int(self.rng.integers(1, 2**31 - 1)),
                "tolfun": 1e-12, "tolx": 1e-10, "tolfunhist": 1e-12}
        self.es = cma.CMAEvolutionStrategy([0.0] * self.dim, self.sigma0, opts)
        self.queue, self.pending, self.pop, self.vals, self.xs = [], {}, [], [], []

    def _to_x(self, u):
        return self.center + self.L @ np.asarray(u, float)

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                if self.pending:
                    break  # wait for current generation
                if self.es is None:
                    self._new_es(self.x0)
                elif self.es.stop():
                    self.n_restarts += 1
                    self._new_es(self.best_x if self.best_x is not None else self.x0)
                self.pop = [np.asarray(p, float) for p in self.es.ask()]
                self.vals = [None] * len(self.pop)
                self.xs = [self._to_x(p) for p in self.pop]
                self.queue = list(range(len(self.pop)))
            i = self.queue.pop(0)
            x = np.clip(self.xs[i], self.lo, self.hi)
            self.pending[tuple(np.round(x, 12))] = i
            out.append([float(v) for v in x])
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = np.asarray(x, float)
            v = float(v)
            if v < self.best_y:
                self.best_y, self.best_x = v, x.copy()
            k = tuple(np.round(x, 12))
            if k in self.pending:
                i = self.pending.pop(k)
                d = self.xs[i] - x
                self.vals[i] = v + self.penalty * float(d @ d)
        if self.es is not None and self.vals and all(v is not None for v in self.vals) \
                and not self.queue and not self.pending:
            self.es.tell(self.pop, self.vals)
            self.pop, self.vals = [], []

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
