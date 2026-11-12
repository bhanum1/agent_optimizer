"""Sequential CMA-ES (pycma) with IPOP restarts and an optional replicate phase.

Args:
  seed: RNG seed.
  sigma0: initial step size (in encoded units).
  popsize: initial population (None = pycma default).
  n_rep: replicates of the best warm-start point asked first (noise estimate).
  x0: optional start point; default = best point told before the first ask.
"""
import pickle
import numpy as np
import cma


class Optimizer:
    max_ask = None
    batch = False

    def __init__(self, space, seed=0, sigma0=2.0, popsize=None, n_rep=0, x0=None,
                 ipop=True):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.dim = len(b)
        self.seed = seed
        self.sigma0 = sigma0
        self.popsize = popsize
        self.n_rep = n_rep
        self.x0 = None if x0 is None else list(x0)
        self.ipop = ipop
        self.rng = np.random.default_rng(seed)
        self.es = None
        self.n_restarts = 0
        self.queue = []       # population points not yet asked
        self.pop = []         # current population (encoded lists)
        self.vals = {}        # index in pop -> y
        self.outstanding = [] # (kind, idx, x) asked, not told
        self.reps_left = None
        self.best_x, self.best_y = None, np.inf
        self.history = []

    # ---- helpers
    def _new_es(self):
        x0 = self.x0 if self.x0 is not None else (
            self.best_x if self.best_x is not None else list((self.lo + self.hi) / 2))
        opts = {"bounds": [list(self.lo), list(self.hi)], "verbose": -9,
                "seed": int(self.rng.integers(1, 2**31 - 1))}
        if self.popsize:
            opts["popsize"] = int(self.popsize * (2 ** self.n_restarts if self.ipop else 1))
        elif self.n_restarts and self.ipop:
            opts["popsize"] = int((4 + int(3 * np.log(self.dim))) * 2 ** self.n_restarts)
        self.es = cma.CMAEvolutionStrategy(list(x0), self.sigma0, opts)

    def _new_pop(self):
        if self.es is None or self.es.stop():
            if self.es is not None:
                self.n_restarts += 1
                self.x0 = list(self.rng.uniform(self.lo, self.hi))
            self._new_es()
        sols = self.es.ask()
        self.pop = [list(np.clip(s, self.lo, self.hi)) for s in sols]
        self.raw = [list(s) for s in sols]
        self.vals = {}
        self.queue = list(range(len(self.pop)))

    def _maybe_tell_es(self):
        if self.pop and len(self.vals) == len(self.pop):
            self.es.tell([np.array(r) for r in self.raw],
                         [self.vals[i] for i in range(len(self.pop))])
            self.pop, self.raw, self.vals, self.queue = [], [], {}, []

    # ---- interface
    def ask(self, n):
        if self.reps_left is None:
            self.reps_left = self.n_rep if self.best_x is not None else 0
        out = []
        for _ in range(n):
            if self.reps_left > 0:
                self.reps_left -= 1
                x = list(self.best_x)
                self.outstanding.append(("rep", -1, x))
                out.append(x)
                continue
            if not self.queue:
                if self.pop:  # population fully asked, waiting for values
                    break
                self._new_pop()
            i = self.queue.pop(0)
            x = [float(v) for v in self.pop[i]]
            self.outstanding.append(("pop", i, x))
            out.append(x)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v)
            self.history.append((list(map(float, x)), v))
            match = None
            for k, (kind, i, xo) in enumerate(self.outstanding):
                if np.allclose(xo, x, atol=1e-9):
                    match = k
                    break
            if match is not None:
                kind, i, _ = self.outstanding.pop(match)
                if kind == "pop" and self.pop:
                    self.vals[i] = v
            elif self.es is None and v < self.best_y:
                self.best_x, self.best_y = list(map(float, x)), v
            if v < self.best_y and match is not None:
                self.best_y = v
        self._maybe_tell_es()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
