"""Scrambled Sobol space-filling design over the encoded box (batch)."""
import json

import numpy as np
from scipy.stats import qmc


class Optimizer:
    batch = True

    def __init__(self, space, seed=0, n=64, **_):
        self.bounds = [list(b) for b in space["bounds"]]
        self.seed = int(seed)
        self.n = int(n)
        self.i = 0

    def _points(self):
        lo, hi = np.array(self.bounds).T
        s = qmc.Sobol(d=len(lo), scramble=True, seed=self.seed)
        return qmc.scale(s.random(self.n), lo, hi).tolist()

    def ask(self, n):
        pts = self._points()[self.i:self.i + n]
        self.i += len(pts)
        return pts

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"bounds": self.bounds, "seed": self.seed, "n": self.n, "i": self.i}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls.__new__(cls)
        o.__dict__.update(d)
        return o
