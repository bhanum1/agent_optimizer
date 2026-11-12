import pickle
import numpy as np
import cma


class Optimizer:
    def __init__(self, space, seed=0, pop0=12, sigma0=2.0, inc=2.0, x0=None):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.dim = space["dim"]
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.pop = pop0
        self.sigma0 = sigma0
        self.inc = inc
        self.restarts = 0
        self.x0 = x0
        self.best = (np.inf, None)
        self.es = None
        self.sols = []      # current generation (raw cma points)
        self.vals = []      # told values for current generation
        self.nasked = 0
        self._new_es()

    def _new_es(self):
        x0 = self.rng.uniform(self.lo, self.hi)
        if self.x0 is not None:
            x0, self.x0 = np.array(self.x0, float), None
        span = self.hi - self.lo
        opts = {"popsize": int(self.pop), "bounds": [self.lo.tolist(), self.hi.tolist()],
                "scaling_of_variables": span / 10.0, "seed": int(self.rng.integers(1, 2**31 - 1)),
                "verbose": -9}
        self.es = cma.CMAEvolutionStrategy(x0.tolist(), self.sigma0, opts)
        self.sols, self.vals, self.nasked = [], [], 0

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.nasked >= len(self.sols):
                if len(self.vals) < len(self.sols):
                    break
                if self.es.stop():
                    self.restarts += 1
                    self.pop = int(self.pop * self.inc)
                    self._new_es()
                self.sols = [np.asarray(s) for s in self.es.ask()]
                self.vals, self.nasked = [], 0
            out.append(np.clip(self.sols[self.nasked], self.lo, self.hi).tolist())
            self.nasked += 1
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            if v < self.best[0]:
                self.best = (float(v), x)
            if len(self.vals) < self.nasked:
                self.vals.append(float(v))
        if self.sols and len(self.vals) == len(self.sols):
            self.es.tell([s.tolist() for s in self.sols], self.vals)
            self.sols, self.vals, self.nasked = [], [], 0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
