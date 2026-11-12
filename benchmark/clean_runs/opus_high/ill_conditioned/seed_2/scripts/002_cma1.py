import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES with IPOP restarts. First run starts at the best known point."""

    def __init__(self, space, seed=0, sigma0=2.0, popsize=None, inc_popsize=2,
                 restart_sigma=2.5, start=None):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.dim = len(self.lo)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.sigma0 = sigma0
        self.popsize = popsize or (4 + int(3 * np.log(self.dim)))
        self.inc_popsize = inc_popsize
        self.restart_sigma = restart_sigma
        self.start = None if start is None else np.array(start, float)
        self.best_x, self.best_y = None, np.inf
        self.es = None
        self.queue = []       # candidates generated, not yet asked
        self.pending = {}     # key -> index in current population
        self.pop, self.vals = [], []
        self.n_restarts = 0

    def _new_es(self, x0, sigma):
        opts = {"bounds": [list(self.lo), list(self.hi)], "popsize": self.popsize,
                "seed": int(self.rng.integers(1, 2**31 - 1)), "verbose": -9,
                "tolfun": 1e-14, "tolx": 1e-13, "tolfunhist": 1e-14}
        self.es = cma.CMAEvolutionStrategy(list(x0), sigma, opts)
        self.pop, self.vals, self.queue, self.pending = [], [], [], {}

    def _ensure_es(self):
        if self.es is None:
            x0 = self.start if self.start is not None else (
                self.best_x if self.best_x is not None else (self.lo + self.hi) / 2)
            self._new_es(np.clip(x0, self.lo, self.hi), self.sigma0)
        elif self.es.stop() and not self.queue and not self.pending:
            self.n_restarts += 1
            self.popsize *= self.inc_popsize
            x0 = self.rng.uniform(self.lo, self.hi)
            self._new_es(x0, self.restart_sigma)

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                if self.pending:
                    break  # wait for current generation
                self._ensure_es()
                self.pop = [np.asarray(p, float) for p in self.es.ask()]
                self.vals = [None] * len(self.pop)
                self.queue = list(range(len(self.pop)))
            i = self.queue.pop(0)
            x = np.clip(self.pop[i], self.lo, self.hi)
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
                self.vals[self.pending.pop(k)] = v
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
