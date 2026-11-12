import numpy as np, pickle
from scipy.stats import qmc

class Optimizer:
    def __init__(self, space, seed=0):
        self.b = np.array(space["bounds"], float)
        self.seed = seed
        self.k = 0

    def _pts(self, n):
        s = qmc.Sobol(len(self.b), scramble=True, seed=self.seed)
        u = s.random(self.k + n)[self.k:]
        self.k += n
        return (self.b[:, 0] + u * (self.b[:, 1] - self.b[:, 0])).tolist()

    def ask(self, n):
        return self._pts(n)

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.b, self.seed, self.k), f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.b, o.seed, o.k = pickle.load(f)
        return o
