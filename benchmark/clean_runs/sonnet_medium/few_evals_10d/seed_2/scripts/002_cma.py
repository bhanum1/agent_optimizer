import pickle
import numpy as np
import cma


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, sigma=1.5, popsize=12):
        self.lo = np.array([b[0] for b in space["bounds"]])
        self.hi = np.array([b[1] for b in space["bounds"]])
        self.seed, self.sigma, self.popsize = seed, sigma, popsize
        self.es = None
        self.hist = []      # (x, y) seen before es started
        self.queue = []     # points from current generation not yet handed out
        self.out = []       # handed out, not told
        self.gen = []       # (x, y) told in current generation
        self.cur = []       # current generation's candidate list

    def _start(self):
        if self.hist:
            h = sorted(self.hist, key=lambda t: t[1])[:5]
            x0 = np.mean([t[0] for t in h], axis=0)
        else:
            x0 = (self.lo + self.hi) / 2
        self.es = cma.CMAEvolutionStrategy(
            list(x0), self.sigma,
            {"bounds": [list(self.lo), list(self.hi)], "popsize": self.popsize,
             "seed": self.seed + 1, "verbose": -9})

    def ask(self, n):
        if self.es is None:
            self._start()
        pts = []
        for _ in range(n):
            if not self.queue:
                self.cur = [np.array(p) for p in self.es.ask()]
                self.queue = list(self.cur)
                self.gen = []
            p = self.queue.pop(0)
            self.out.append(p)
            pts.append([float(v) for v in p])
        return pts

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = np.array(x, float)
            m = [i for i, o in enumerate(self.out) if np.allclose(o, x)]
            if m and self.es is not None:
                self.out.pop(m[0])
                self.gen.append((x, float(v)))
                if len(self.gen) == len(self.cur):
                    self.es.tell([t[0] for t in self.gen], [t[1] for t in self.gen])
                    self.gen, self.cur = [], []
            elif self.es is None:
                self.hist.append((x, float(v)))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
