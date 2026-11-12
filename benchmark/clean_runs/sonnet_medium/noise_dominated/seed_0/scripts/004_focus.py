import numpy as np, pickle

class Optimizer:
    def __init__(self, space, seed=0, k=40, scale=0.8, refit=25):
        self.lo = np.array([b[0] for b in space["bounds"]]); self.hi = np.array([b[1] for b in space["bounds"]])
        self.k, self.scale, self.refit = k, scale, refit
        self.rng = np.random.default_rng(seed)
        self.X, self.y = [], []
        self.mu = np.array([0.2, -0.8, -0.2, 2.3, -2.3]); self.cov = np.diag([1.0] * 5)
        self.n = 0

    def _refit(self):
        if len(self.y) < self.k: return
        idx = np.argsort(self.y)[: self.k]
        P = np.array(self.X)[idx]
        self.mu = P.mean(0)
        self.cov = np.cov(P.T) * self.scale ** 2 + 0.05 * np.eye(5)

    def ask(self, n):
        out = []
        for _ in range(n):
            for _ in range(100):
                x = self.rng.multivariate_normal(self.mu, self.cov)
                if np.all(x >= self.lo) and np.all(x <= self.hi): break
            else:
                x = np.clip(x, self.lo, self.hi)
            out.append([float(v) for v in x])
        return out

    def tell(self, X, y, g):
        self.X += [list(x) for x in X]; self.y += list(y)
        self.n += len(y)
        self._refit()

    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f: return pickle.load(f)
