import pickle
import numpy as np
from scipy.stats import qmc


class Optimizer:
    def __init__(self, space, seed=0, **args):
        self.bounds = np.array(space["bounds"], float)
        self.seed = seed
        self.i = 0
        self.sob = qmc.Sobol(d=len(self.bounds), scramble=True, seed=seed)

    def ask(self, n):
        u = self.sob.random(n)
        self.i += n
        X = self.bounds[:, 0] + u * (self.bounds[:, 1] - self.bounds[:, 0])
        return X.tolist()

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
