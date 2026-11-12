import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES with optional fixed coords (none by default): coords in `fixed` are held, the rest are searched
    by pycma (bounded), one point at a time. Restarts with a fresh sigma0 from the best point
    seen if CMA stops."""
    max_ask = 1

    def __init__(self, space, seed=0, x0=None, sigma0=0.3, popsize=10, fixed=None):
        b = np.array(space["bounds"], dtype=float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.d = len(self.lo)
        fixed = fixed or {}
        self.fixed = {int(k): float(v) for k, v in fixed.items()}
        self.free = [i for i in range(self.d) if i not in self.fixed]
        self.sigma0, self.popsize, self.seed = sigma0, popsize, seed
        start = np.asarray(x0, float)[self.free] if x0 is not None else \
            (self.lo[self.free] + self.hi[self.free]) / 2
        self.best_x, self.best_y = start, np.inf
        self.n_restarts = 0
        self._new_es(start)
        self.queue = []      # free-coord solutions from the current generation not yet asked
        self.pending = {}    # key -> free-coord solution asked but not told
        self.gen_X, self.gen_y = [], []
        self.history = []

    def _new_es(self, start):
        self.es = cma.CMAEvolutionStrategy(
            list(start), self.sigma0,
            {"bounds": [list(self.lo[self.free]), list(self.hi[self.free])],
             "popsize": self.popsize, "seed": self.seed + 1000 * self.n_restarts + 1,
             "verbose": -9})

    def _full(self, z):
        x = np.empty(self.d)
        for i, v in self.fixed.items():
            x[i] = v
        x[self.free] = z
        return [float(v) for v in np.clip(x, self.lo, self.hi)]

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                if self.pending:
                    break  # wait for the rest of this generation
                if self.es.stop():
                    self.n_restarts += 1
                    self._new_es(self.best_x)
                self.queue = [np.asarray(z) for z in self.es.ask()]
            z = self.queue.pop(0)
            x = self._full(z)
            self.pending[tuple(np.round(x, 12))] = z
            out.append(x)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v) if np.isfinite(v) else 1e300
            self.history.append((list(map(float, x)), v))
            key = tuple(np.round(np.asarray(x, float), 12))
            z = self.pending.pop(key, None)
            if z is None:
                continue  # not one of ours (e.g. warm start data)
            if v < self.best_y:
                self.best_y, self.best_x = v, np.asarray(z)
            self.gen_X.append(z); self.gen_y.append(v)
        if not self.pending and not self.queue and self.gen_X:
            self.es.tell(self.gen_X, self.gen_y)
            self.gen_X, self.gen_y = [], []

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
