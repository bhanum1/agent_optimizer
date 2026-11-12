import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES with IPOP restarts. First run starts at the best known point
    (from warm-start data) unless x0 is given; restarts double popsize and
    alternate between a random start and the incumbent best."""
    max_ask = None
    batch = False

    def __init__(self, space, sigma0=2.0, seed=0, popsize=None, x0=None,
                 inc_popsize=2, tolfun=1e-11):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.dim = len(self.lo)
        self.sigma0 = float(sigma0)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.popsize = popsize
        self.x0 = None if x0 is None else list(map(float, x0))
        self.inc = inc_popsize
        self.tolfun = tolfun
        self.es = None
        self.restarts = 0
        self.queue = []        # points of the current population not yet asked
        self.pop = []          # current population (list of lists)
        self.vals = {}         # index in pop -> value
        self.best_x, self.best_y = None, np.inf
        self.history = []

    def _key(self, x):
        return tuple(np.round(np.asarray(x, float), 12))

    def _new_es(self):
        if self.es is None and self.x0 is not None:
            x0 = np.array(self.x0)
        elif self.es is None and self.best_x is not None:
            x0 = np.array(self.best_x)
        elif self.restarts % 2 == 1 or self.best_x is None:
            x0 = self.lo + self.rng.random(self.dim) * (self.hi - self.lo)
        else:
            x0 = np.array(self.best_x)
        opts = {"bounds": [list(self.lo), list(self.hi)],
                "seed": int(self.rng.integers(1, 2**31 - 1)),
                "verbose": -9, "tolfun": self.tolfun, "tolx": 1e-12,
                "tolfunhist": 1e-13}
        ps = self.popsize
        if ps is None:
            ps = 4 + int(3 * np.log(self.dim))
        opts["popsize"] = int(ps * self.inc ** self.restarts)
        # random restarts use the full step size, incumbent restarts half of it
        sig = self.sigma0 if self.restarts % 2 == 1 or self.best_x is None else self.sigma0 / 2
        if self.restarts == 0:
            sig = self.sigma0
        self.es = cma.CMAEvolutionStrategy(list(x0), sig, opts)
        self._new_pop()

    def _new_pop(self):
        sols = self.es.ask()
        self.pop = [list(map(float, np.clip(s, self.lo, self.hi))) for s in sols]
        self.queue = list(range(len(self.pop)))
        self.vals = {}

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.es is None:
                self._new_es()
            if not self.queue:
                # population fully asked but not told: sample an extra point
                # from the current distribution (rare; only if n exceeds popsize)
                s = self.es.ask(1)[0]
                out.append(list(map(float, np.clip(s, self.lo, self.hi))))
                continue
            i = self.queue.pop(0)
            out.append(self.pop[i])
        return out

    def tell(self, X, y, g):
        keys = {self._key(p): i for i, p in enumerate(self.pop)}
        for x, v in zip(X, y):
            v = float(v)
            if not np.isfinite(v):
                v = 1e300
            self.history.append((list(map(float, x)), v))
            if v < self.best_y:
                self.best_y, self.best_x = v, list(map(float, x))
            i = keys.get(self._key(x))
            if i is not None and i not in self.vals and self.es is not None:
                self.vals[i] = v
        if self.es is not None and self.pop and len(self.vals) == len(self.pop):
            self.es.tell([np.array(p) for p in self.pop],
                         [self.vals[i] for i in range(len(self.pop))])
            if self.es.stop():
                self.restarts += 1
                self._new_es()
            else:
                self._new_pop()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
