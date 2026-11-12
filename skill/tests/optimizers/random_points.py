"""Test optimizer: uniform random points in the box. Does not adapt, so a block is requested at once."""
import pickle
import random


class Optimizer:
    batch = True

    def __init__(self, space, seed=0):
        self.bounds, self.rng = space["bounds"], random.Random(seed)

    def ask(self, n):
        return [[self.rng.uniform(lo, hi) for lo, hi in self.bounds] for _ in range(n)]

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
