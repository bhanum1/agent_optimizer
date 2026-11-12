import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES with IPOP restarts (population doubles at each restart)."""

    def __init__(self, space, seed=0, sigma0=0.3, pop0=None, growth=2.0):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lo)
        self.rng = np.random.RandomState(seed)
        self.sigma0 = sigma0
        self.pop = pop0 or (4 + int(3 * np.log(self.d)))
        self.growth = growth
        self.queue = []
        self.told = []
        self.sols = []
        self.es = None
        self._new_es()

    def _new_es(self):
        x0 = self.lo + self.rng.rand(self.d) * (self.hi - self.lo)
        s = self.sigma0 * (self.hi - self.lo).mean()
        self.es = cma.CMAEvolutionStrategy(
            list(x0), s,
            {"popsize": int(self.pop), "bounds": [list(self.lo), list(self.hi)],
             "seed": int(self.rng.randint(1, 2**30)), "verbose": -9})
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
                if self.es.stop():
                    self.pop *= self.growth
                    self._new_es()
                else:
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
