"""CMA-ES (pycma) local search with ask/tell one point at a time.

args: x0 (start, user units; default box centre), sigma0 (user units), popsize,
seed. Foreign points (warm start) are only recorded; if x0 is None and warm data
is told before the first ask, the best told point becomes the start.
Restarts (same x0 = current best, sigma0) when CMA stops.
"""
import pickle
import numpy as np
import cma


class Optimizer:
    def __init__(self, space, x0=None, sigma0=0.5, popsize=None, seed=0):
        self.lb = np.array([b[0] for b in space["bounds"]], float)
        self.ub = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lb)
        self.x0 = None if x0 is None else np.asarray(x0, float)
        self.sigma0 = float(sigma0)
        self.popsize = popsize
        self.seed = int(seed)
        self.es = None
        self.queue = []      # points of current generation not yet asked
        self.gen = []        # points of current generation asked (in order)
        self.vals = {}       # index in gen -> value
        self.pending = []
        self.best_x, self.best_y = None, np.inf
        self.n_restarts = 0

    def _start(self):
        x0 = self.x0 if self.x0 is not None else (
            self.best_x if self.best_x is not None else (self.lb + self.ub) / 2)
        opts = {"bounds": [self.lb.tolist(), self.ub.tolist()],
                "seed": self.seed + 1000 * self.n_restarts, "verbose": -9}
        if self.popsize:
            opts["popsize"] = int(self.popsize)
        self.es = cma.CMAEvolutionStrategy(np.asarray(x0).tolist(), self.sigma0, opts)

    def _new_gen(self):
        if self.es is None:
            self._start()
        if self.es.stop():
            self.n_restarts += 1
            self.x0 = self.best_x
            self._start()
        self.queue = [np.asarray(x, float) for x in self.es.ask()]
        self.gen = []
        self.vals = {}

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                if self.gen and len(self.vals) < len(self.gen):
                    # generation not complete: resample extra point from same gen
                    raise RuntimeError("generation incomplete")
                self._new_gen()
            x = np.clip(self.queue.pop(0), self.lb, self.ub)
            self.gen.append(x)
            self.pending.append(x)
            out.append(x.tolist())
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = np.asarray(x, float)
            v = float(v)
            if v < self.best_y:
                self.best_y, self.best_x = v, x.copy()
            k = next((i for i, p in enumerate(self.pending) if np.allclose(p, x, atol=1e-9)), None)
            if k is None:
                continue
            self.pending.pop(k)
            j = next(i for i, p in enumerate(self.gen)
                     if i not in self.vals and np.allclose(p, x, atol=1e-9))
            self.vals[j] = v
        if self.gen and not self.queue and len(self.vals) == len(self.gen):
            self.es.tell([list(p) for p in self.gen], [self.vals[i] for i in range(len(self.gen))])
            self.gen = []
            self.vals = {}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
