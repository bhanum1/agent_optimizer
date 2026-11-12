import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES started at a given point with a small sigma (local refinement)."""

    def __init__(self, space, x0=None, sigma=0.02, pop=8, seed=0):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.es = cma.CMAEvolutionStrategy(
            list(x0), sigma,
            {"popsize": int(pop), "bounds": [list(self.lo), list(self.hi)],
             "seed": int(seed) + 1, "verbose": -9})
        self.queue = []
        self.told = []
        self.sols = []
        self._next_gen()

    def _next_gen(self):
        self.sols = [np.array(s) for s in self.es.ask()]
        self.queue = list(self.sols)
        self.told = []

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                break
            out.append([float(v) for v in np.clip(self.queue.pop(0), self.lo, self.hi)])
        return out

    def tell(self, X, y, g):
        for v in y:
            self.told.append(float(v))
            if len(self.told) == len(self.sols):
                self.es.tell([list(s) for s in self.sols], self.told)
                self._next_gen()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
