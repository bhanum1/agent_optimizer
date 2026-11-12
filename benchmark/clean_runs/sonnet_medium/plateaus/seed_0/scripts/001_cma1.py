import pickle
import numpy as np
import cma
from scipy.stats import qmc


class Optimizer:
    def __init__(self, space, seed=0, n_sobol=50, sigma0=0.3, pop0=8):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lo)
        self.seed, self.sigma0, self.pop = seed, sigma0, pop0
        self.rng = np.random.RandomState(seed)
        self.sobol = [list(self.lo + p * (self.hi - self.lo))
                      for p in qmc.Sobol(self.d, seed=seed).random(n_sobol)]
        self.hist = []          # (x, y) every told point
        self.es = None
        self.batch = []         # current cma solutions
        self.vals = {}          # index in batch -> value
        self.next_i = 0
        self.stall = 0
        self.best_run = np.inf

    def _new_es(self):
        if self.hist and self.rng.rand() < 0.5:
            xs = np.array([h[0] for h in self.hist]); ys = np.array([h[1] for h in self.hist])
            x0 = xs[np.argsort(ys)[self.rng.randint(min(5, len(ys)))]]
            sig = self.sigma0 * 0.5
        else:
            x0 = self.lo + self.rng.rand(self.d) * (self.hi - self.lo)
            sig = self.sigma0
        s = (self.hi - self.lo)
        self.es = cma.CMAEvolutionStrategy(
            list((x0 - self.lo) / s), sig,
            {"bounds": [0, 1], "popsize": int(self.pop), "seed": int(self.rng.randint(1, 10**6)),
             "verbose": -9, "tolfun": 1e-12, "tolx": 1e-10})
        self.batch, self.vals, self.next_i, self.best_run = [], {}, 0, np.inf

    def _refill(self):
        if self.es is None or self.es.stop():
            if self.es is not None:
                self.pop = min(self.pop * 1.5, 40)
            self._new_es()
        self.batch = [np.array(x) for x in self.es.ask()]
        self.vals, self.next_i = {}, 0

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.sobol:
                out.append(self.sobol.pop(0)); continue
            if self.next_i >= len(self.batch):
                if self.batch and len(self.vals) < len(self.batch):
                    # waiting on results; re-propose an outstanding point
                    out.append(list(self.lo + self.batch[-1] * (self.hi - self.lo))); continue
                self._refill()
            x = self.batch[self.next_i]; self.next_i += 1
            out.append(list(self.lo + x * (self.hi - self.lo)))
        return out

    def tell(self, X, y, g):
        s = self.hi - self.lo
        for x, v in zip(X, y):
            x = np.array(x, float); self.hist.append((x, float(v)))
            u = (x - self.lo) / s
            for i, b in enumerate(self.batch):
                if i not in self.vals and np.allclose(b, u, atol=1e-9):
                    self.vals[i] = float(v); break
        if self.batch and len(self.vals) == len(self.batch):
            self.es.tell([list(b) for b in self.batch], [self.vals[i] for i in range(len(self.batch))])
            self.batch, self.vals, self.next_i = [], {}, 0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
