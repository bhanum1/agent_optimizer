import numpy as np, pickle

class Optimizer:
    """(1+1) local search accepting ties; step scale drawn log-uniformly so plateaus are crossed."""
    def __init__(self, space, seed=0, lo=0.05, hi=1.0):
        self.b = np.array(space["bounds"], float)
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = lo, hi
        self.best_x = None
        self.best_y = np.inf
        self.pending = []

    def ask(self, n):
        pts = []
        for _ in range(n):
            if self.best_x is None:
                p = self.rng.uniform(self.b[:, 0], self.b[:, 1])
            else:
                s = np.exp(self.rng.uniform(np.log(self.lo), np.log(self.hi)))
                d = self.rng.normal(size=len(self.b))
                m = self.rng.random(len(self.b)) < 0.6
                if not m.any():
                    m[self.rng.integers(len(self.b))] = True
                p = np.clip(self.best_x + s * d * m, self.b[:, 0], self.b[:, 1])
            pts.append(p.tolist())
        return pts

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            if v <= self.best_y:
                self.best_y, self.best_x = float(v), np.array(x, float)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
