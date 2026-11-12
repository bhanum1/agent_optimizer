import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES (pycma) asked one point at a time; a generation is told once all its points return.
    Tells of points it did not ask (e.g. warm start) are ignored."""
    max_ask = None
    batch = False

    def __init__(self, space, x0=None, sigma0=0.1, popsize=10, seed=0):
        b = np.array(space["bounds"], dtype=float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        if x0 is None:
            x0 = (self.lo + self.hi) / 2
        opts = {"bounds": [list(self.lo), list(self.hi)], "popsize": popsize,
                "seed": seed, "verbose": -9, "tolfun": 0, "tolx": 1e-12,
                "tolfunhist": 0, "tolstagnation": 10**9}
        self.es = cma.CMAEvolutionStrategy(list(map(float, x0)), sigma0, opts)
        self.pop = []       # current generation (list of lists)
        self.vals = {}      # index -> y
        self.next = 0       # next index of pop to hand out
        self.gens = 0

    def _new_gen(self):
        self.pop = [list(map(float, np.clip(p, self.lo, self.hi))) for p in self.es.ask()]
        self.vals = {}
        self.next = 0

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.pop or self.next >= len(self.pop):
                if self.pop and len(self.vals) < len(self.pop):
                    break  # waiting on outstanding points of this generation
                self._new_gen()
            out.append(self.pop[self.next])
            self.next += 1
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = list(map(float, x))
            for i in range(self.next):
                if i not in self.vals and np.allclose(self.pop[i], x, atol=1e-12):
                    self.vals[i] = float(v)
                    break
        if self.pop and len(self.vals) == len(self.pop):
            vals = [self.vals[i] if np.isfinite(self.vals[i]) else 1e300 for i in range(len(self.pop))]
            self.es.tell([np.array(p) for p in self.pop], vals)
            self.gens += 1
            self.pop, self.vals, self.next = [], {}, 0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
