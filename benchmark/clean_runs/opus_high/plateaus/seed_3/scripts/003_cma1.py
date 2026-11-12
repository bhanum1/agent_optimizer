import pickle
import numpy as np
import cma


class Optimizer:
    """IPOP CMA-ES. Starts at x0 (or best warm-start point), restarts with doubled
    population when the run stops (incl. flat fitness on plateaus). Ties broken
    randomly by a tiny jitter on y so plateaus do not freeze the ranking."""

    def __init__(self, space, x0=None, sigma0=1.0, popsize=10, seed=0, max_pop=80):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.x0 = None if x0 is None else list(map(float, x0))
        self.sigma0, self.popsize, self.seed, self.max_pop = float(sigma0), int(popsize), int(seed), int(max_pop)
        self.rng = np.random.default_rng(seed)
        self.es = None
        self.pending = []      # solutions asked, not yet told, in order
        self.results = {}      # key -> y
        self.best_x, self.best_y = None, np.inf
        self.restarts = 0

    def _new_es(self):
        if self.restarts == 0 and self.x0 is not None:
            x0 = self.x0
        elif self.best_x is not None:
            x0 = list(self.best_x)
        else:
            x0 = list((self.lo + self.hi) / 2)
        opts = {"bounds": [list(self.lo), list(self.hi)], "popsize": self.popsize,
                "seed": int(self.rng.integers(1, 2**31 - 1)), "verbose": -9,
                "tolflatfitness": 3, "tolfun": 1e-9, "tolx": 1e-6}
        self.es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
        self.gen = []

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.es is None or (not self.gen and not self.pending and self.es.stop()):
                if self.es is not None:
                    self.restarts += 1
                    self.popsize = min(self.max_pop, self.popsize * 2)
                self._new_es()
            if not self.gen:
                if self.pending:      # generation not complete: extra sample, not ranked
                    x = list(map(float, np.clip(self.es.mean + self.es.sigma * self.rng.standard_normal(len(self.lo)), self.lo, self.hi)))
                    self.pending.append(x)
                    out.append(x)
                    continue
                self.gen = [list(map(float, s)) for s in self.es.ask()]
                self.cur = list(self.gen)
            x = self.gen.pop(0)
            self.pending.append(x)
            out.append(x)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = list(map(float, x)); v = float(v)
            if v < self.best_y:
                self.best_y, self.best_x = v, x
            self.results[tuple(np.round(x, 12))] = v
            if x in self.pending:
                self.pending.remove(x)
        if self.es is not None and not self.gen and not self.pending and getattr(self, "cur", None):
            ys = [self.results[tuple(np.round(s, 12))] for s in self.cur]
            jit = self.rng.uniform(0, 1e-9, len(ys)) * (1 + np.abs(ys))
            self.es.tell([np.array(s) for s in self.cur], list(np.array(ys) + jit))
            self.cur = []

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
