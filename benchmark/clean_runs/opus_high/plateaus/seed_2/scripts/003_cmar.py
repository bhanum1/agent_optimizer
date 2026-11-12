import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES with restarts, one point at a time.

    Restart 0 starts at `x0` (or the best told point) with `sigma_local`.
    Later restarts alternate: odd -> around the global best with `sigma_local`,
    even -> uniform random start with `sigma_global` and doubled popsize (IPOP).
    """

    def __init__(self, space, seed=0, x0=None, sigma_local=0.5, sigma_global=2.5,
                 popsize=None, tolflat=5, stall=80):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.d = len(self.lo)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.x0 = x0
        self.sl, self.sg = sigma_local, sigma_global
        self.pop0 = popsize or (4 + int(3 * np.log(self.d)))
        self.tolflat = tolflat
        self.stall = stall
        self.since = 0      # evaluations since this run last improved
        self.best_x, self.best_y = None, np.inf
        self.k = 0          # restart counter
        self.n_global = 0
        self.es = None
        self.pop, self.vals, self.issued = [], {}, 0
        self.hist = []      # (restart, best y of that restart)
        self.run_best = np.inf

    def _new_es(self):
        if self.k == 0:
            start = np.array(self.x0, float) if self.x0 is not None else (
                self.best_x if self.best_x is not None else (self.lo + self.hi) / 2)
            sig, pop = self.sl, self.pop0
        elif self.k % 2 == 1:
            start, sig, pop = self.best_x, self.sl, self.pop0
        else:
            self.n_global += 1
            start = self.rng.uniform(self.lo, self.hi)
            sig, pop = self.sg, self.pop0 * 2 ** self.n_global
        opts = {"bounds": [list(self.lo), list(self.hi)], "popsize": int(pop),
                "seed": int(self.rng.integers(1, 2**31 - 1)), "verbose": -9,
                "tolflatfitness": self.tolflat, "tolx": 1e-6}
        self.es = cma.CMAEvolutionStrategy(list(start), sig, opts)
        self.run_best = np.inf
        self.since = 0

    def _next_gen(self):
        if self.es is None or self.es.stop() or self.since >= self.stall:
            if self.es is not None:
                self.hist.append((self.k, float(self.run_best)))
                self.k += 1
            self._new_es()
        self.pop = [np.clip(np.array(p), self.lo, self.hi) for p in self.es.ask()]
        self.vals, self.issued = {}, 0

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.issued >= len(self.pop):
                if len(self.vals) < len(self.pop):
                    break   # waiting for outstanding results
                self._next_gen()
            out.append([float(v) for v in self.pop[self.issued]])
            self.issued += 1
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = np.array(x, float)
            v = float(v)
            if v < self.best_y:
                self.best_y, self.best_x = v, x.copy()
            for i in range(self.issued):
                if i not in self.vals and np.allclose(self.pop[i], x, atol=1e-12):
                    self.vals[i] = v
                    if v < self.run_best - 1e-9:
                        self.run_best, self.since = v, 0
                    else:
                        self.since += 1
                    break
        if self.pop and len(self.vals) == len(self.pop):
            self.es.tell([list(p) for p in self.pop],
                         [self.vals[i] for i in range(len(self.pop))])
            self.pop, self.vals, self.issued = [], {}, 0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
