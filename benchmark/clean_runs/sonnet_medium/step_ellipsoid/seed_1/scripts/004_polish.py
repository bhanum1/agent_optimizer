import pickle
import numpy as np
import cma


class Optimizer:
    """Small-sigma CMA-ES from a given start (user units), with restarts at the same centre."""
    max_ask = 1

    def __init__(self, space, x0, sigma0=0.03, popsize=8, seed=0):
        self.b = np.array(space["bounds"], float)
        self.span = self.b[:, 1] - self.b[:, 0]
        self.x0 = ((np.array(x0, float) - self.b[:, 0]) / self.span).tolist()
        self.sigma0, self.pop, self.seed, self.n = sigma0, popsize, seed, 0
        self.best = (np.inf, self.x0)
        self.es = None
        self.cur, self.res, self.queue = [], [], []

    def _gen(self):
        if self.es is None or self.es.stop():
            self.n += 1
            self.es = cma.CMAEvolutionStrategy(
                list(self.best[1]), self.sigma0,
                {"bounds": [0, 1], "popsize": self.pop, "seed": self.seed + self.n,
                 "verbose": -9, "tolfun": 1e-12, "tolx": 1e-12})
        self.cur = [list(map(float, x)) for x in self.es.ask()]
        self.res = []
        self.queue = list(self.cur)

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                self._gen()
            u = self.queue.pop(0)
            out.append((self.b[:, 0] + np.array(u) * self.span).tolist())
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            u = (np.array(x) - self.b[:, 0]) / self.span
            if v < self.best[0]:
                self.best = (float(v), u.copy())
            if self.cur and len(self.res) < len(self.cur):
                self.res.append((self.cur[len(self.res)], float(v)))
                if len(self.res) == len(self.cur):
                    self.es.tell([r[0] for r in self.res], [r[1] for r in self.res])
                    self.cur, self.res = [], []

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
