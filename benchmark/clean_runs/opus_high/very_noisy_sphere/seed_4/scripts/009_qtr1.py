# Quantile-quadratic trust region: fit a quadratic to the lower quantile of y near the
# current center, sample around its (trust-limited) argmin with Gaussian spread.
import pickle
import numpy as np
from scipy.optimize import linprog, minimize


class Optimizer:
    def __init__(self, space, center, radius=0.3, tau=0.2, sigma=0.05, step=0.12,
                 refit=10, seed=0):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.center = np.array(center, float)
        self.target = self.center.copy()
        self.radius, self.tau, self.sigma, self.step, self.refit = radius, tau, sigma, step, refit
        self.X, self.y = [], []
        self.since = 0
        self.rng = np.random.default_rng(seed)
        self.log = []

    def _fit(self):
        if len(self.y) < 42:
            return
        X = np.array(self.X); y = np.array(self.y)
        Z = X - self.center; s = np.linalg.norm(Z, axis=1) < self.radius
        Z, yy = Z[s], y[s]
        if len(yy) < 42:
            return
        iu = np.triu_indices(5)
        F = np.hstack([np.ones((len(Z), 1)), Z, np.array([np.outer(z, z)[iu] for z in Z])])
        n, p = F.shape
        if n < 2 * p:
            return
        cost = np.r_[np.zeros(p), self.tau * np.ones(n), (1 - self.tau) * np.ones(n)]
        A = np.hstack([F, np.eye(n), -np.eye(n)])
        res = linprog(cost, A_eq=A, b_eq=yy, bounds=[(None, None)] * p + [(0, None)] * (2 * n),
                      method="highs")
        if not res.success:
            return
        b = res.x[:p]; g = b[1:6]; H = np.zeros((5, 5)); k = 0
        for i, j in zip(*iu):
            H[i, j] = H[j, i] = b[6 + k] * (1 if i == j else 0.5); k += 1
        f = lambda z: b[0] + g @ z + z @ H @ z
        con = [{"type": "ineq", "fun": lambda z: self.step ** 2 - z @ z}]
        best = min((minimize(f, self.rng.normal(0, 0.03, 5), constraints=con) for _ in range(6)),
                   key=lambda o: o.fun)
        self.target = np.clip(self.center + best.x, self.lo, self.hi)
        self.log.append((len(self.y), float(best.fun), self.target.tolist()))
        # move the data window halfway toward the model's optimum
        self.center = 0.5 * (self.center + self.target)

    def ask(self, n):
        if self.since >= self.refit or not self.log:
            self._fit(); self.since = 0
        out = []
        for _ in range(n):
            out.append(np.clip(self.target + self.rng.normal(0, self.sigma, 5), self.lo, self.hi).tolist())
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append(list(map(float, xi))); self.y.append(float(yi))
        self.since += len(y)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as fh:
            o.__dict__.update(pickle.load(fh))
        return o
