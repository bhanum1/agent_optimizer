import numpy as np, pickle

class Optimizer:
    """(1+1) local search accepting ties; step scale drawn log-uniformly so plateaus are crossed."""
    def __init__(self, space, seed=0, lo=0.05, hi=1.0, starts=(), per=80):
        self.b = np.array(space["bounds"], float)
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = lo, hi
        self.best_x = None
        self.best_y = np.inf
        self.pending = []
        self.starts = [list(map(float, p)) for p in starts]
        self.per, self.cnt, self.k = per, 0, 0
        self.gbest_x, self.gbest_y = None, np.inf

    def ask(self, n):
        pts = []
        for _ in range(n):
            self.cnt += 1
            if self.k < len(self.starts) and self.cnt > self.per:
                self.k, self.cnt = self.k + 1, 1
                if self.k < len(self.starts):
                    self.best_x, self.best_y = None, np.inf
                else:
                    self.best_x, self.best_y = self.gbest_x.copy(), self.gbest_y
            if self.best_x is None and self.k < len(self.starts):
                p = np.array(self.starts[self.k])
            elif self.best_x is None:
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
            if v < self.gbest_y:
                self.gbest_y, self.gbest_x = float(v), np.array(x, float)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
