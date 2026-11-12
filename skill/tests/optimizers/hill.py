"""Test optimizer: Gaussian steps around the best point told so far (adapts within a block)."""
import math
import pickle
import random


class Optimizer:
    def __init__(self, space, seed=0, step=0.2, one_at_a_time=False):
        self.bounds, self.step, self.rng = space["bounds"], step, random.Random(seed)
        self.best, self.f_best, self.n_told = None, math.inf, 0
        if one_at_a_time:
            self.max_ask = 1

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.best is None:
                out.append([self.rng.uniform(lo, hi) for lo, hi in self.bounds])
            else:
                out.append([min(hi, max(lo, b + self.rng.gauss(0, self.step * (hi - lo))))
                            for b, (lo, hi) in zip(self.best, self.bounds)])
        return out

    def tell(self, X, y, g):
        self.n_told += len(y)
        for x, v in zip(X, y):
            if v < self.f_best:
                self.best, self.f_best = list(x), v

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
