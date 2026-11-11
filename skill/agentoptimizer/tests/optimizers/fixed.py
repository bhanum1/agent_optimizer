"""Test optimizer: the encoded points in `points`, then finishes."""
import pickle


class Optimizer:
    batch = True

    def __init__(self, space, points=(), seed=0):
        self.points = [list(p) for p in points]

    def ask(self, n):
        out, self.points = self.points[:n], self.points[n:]
        return out

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
