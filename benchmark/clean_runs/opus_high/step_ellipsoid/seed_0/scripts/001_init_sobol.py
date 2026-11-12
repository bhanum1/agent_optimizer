import json

import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Scrambled Sobol space-filling design over the encoded box."""

    def __init__(self, space, seed=0, n=64):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.seed = int(seed)
        self.n = int(n)
        self.i = 0

    def _points(self):
        s = qmc.Sobol(d=len(self.bounds), scramble=True, seed=self.seed)
        u = s.random(self.n)
        return qmc.scale(u, self.bounds[:, 0], self.bounds[:, 1])

    def ask(self, n):
        if self.i >= self.n:
            return []
        pts = self._points()[self.i:self.i + n]
        self.i += len(pts)
        return [list(map(float, p)) for p in pts]

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"bounds": self.bounds.tolist(), "seed": self.seed,
                       "n": self.n, "i": self.i}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls.__new__(cls)
        o.bounds = np.array(d["bounds"])
        o.seed, o.n, o.i = d["seed"], d["n"], d["i"]
        return o
