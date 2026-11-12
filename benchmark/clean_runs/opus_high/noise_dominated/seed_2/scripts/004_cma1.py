import pickle
import warnings

import numpy as np

warnings.filterwarnings("ignore")


class Optimizer:
    """CMA-ES (pycma) handing out one generation point by point; rank-based, so
    upward noise spikes only demote single samples. IPOP-style restarts from the
    best measured point when the run stops."""

    def __init__(self, space, x0=None, sigma0=2.0, popsize=12, seed=0,
                 min_sigma=0.02, inc_popsize=2):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        d = len(self.lo)
        self.x0 = list(x0) if x0 is not None else list((self.lo + self.hi) / 2)
        self.sigma0 = sigma0
        self.popsize = popsize
        self.seed = seed
        self.min_sigma = min_sigma
        self.inc_popsize = inc_popsize
        self.restarts = 0
        self.best = (np.inf, None)
        self.pending = []      # points of the current generation not yet handed out
        self.gen_X, self.gen_y = [], []   # told results for the current generation
        self.gen_n = 0
        self._new_es(self.x0)

    def _new_es(self, x0):
        import cma
        opts = {"bounds": [list(self.lo), list(self.hi)], "popsize": self.popsize,
                "seed": int(self.seed + 1000 * self.restarts + 1), "verbose": -9,
                "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 1000,
                "tolx": self.min_sigma * 0.1}
        self.es = cma.CMAEvolutionStrategy(list(x0), self.sigma0, opts)
        self.pending, self.gen_X, self.gen_y = [], [], []

    def _new_gen(self):
        if self.es.stop() or self.es.sigma * np.max(np.sqrt(self.es.dC)) < self.min_sigma:
            self.restarts += 1
            self.popsize = int(self.popsize * self.inc_popsize)
            x0 = self.best[1] if self.best[1] is not None else self.x0
            self._new_es(x0)
        sols = self.es.ask()
        self.pending = [list(map(float, np.clip(s, self.lo, self.hi))) for s in sols]
        self.gen_n = len(self.pending)
        self.gen_X, self.gen_y = [], []

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.pending and len(self.gen_y) >= self.gen_n:
                self._new_gen()
            if not self.pending:
                # generation fully handed out but not all told yet
                break
            out.append(self.pending.pop(0))
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v)
            if not np.isfinite(v):
                v = max(self.gen_y + [1e6])
            x = [float(a) for a in x]
            if v < self.best[0]:
                self.best = (v, x)
            self.gen_X.append(x)
            self.gen_y.append(v)
        # warm-start points (not from this generation) only update the best
        if self.gen_n and len(self.gen_y) >= self.gen_n:
            if len(self.gen_y) == self.gen_n:
                self.es.tell([np.array(a) for a in self.gen_X], self.gen_y)
            else:
                # mixed with foreign points: tell the last gen_n (this generation)
                self.es.tell([np.array(a) for a in self.gen_X[-self.gen_n:]],
                             self.gen_y[-self.gen_n:])
            self.gen_X, self.gen_y = [], []
            self.gen_n = 0
        elif not self.gen_n:
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
