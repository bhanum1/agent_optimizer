import json

import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Scrambled Sobol space-filling design over the box."""

    batch = True

    def __init__(self, space, seed=0, n=64):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.seed = seed
        self.n = n
        self.issued = 0

    def _points(self):
        s = qmc.Sobol(d=len(self.bounds), scramble=True, seed=self.seed)
        u = s.random(self.n)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + u * (hi - lo)

    def ask(self, n):
        pts = self._points()[self.issued:self.issued + n]
        self.issued += len(pts)
        return [list(map(float, p)) for p in pts]

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"bounds": self.bounds.tolist(), "seed": self.seed,
                       "n": self.n, "issued": self.issued}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls({"bounds": d["bounds"]}, seed=d["seed"], n=d["n"])
        o.issued = d["issued"]
        return o
