import pickle
import numpy as np
from scipy.optimize import minimize


def _feat(Z):
    Z = np.atleast_2d(Z)
    d = Z.shape[1]
    cols = [np.ones(len(Z))] + [Z[:, i] for i in range(d)]
    cols += [Z[:, i] * Z[:, j] for i in range(d) for j in range(i, d)]
    return np.array(cols).T


class Optimizer:
    """Quadratic-model trust region around the incumbent.

    Each ask fits a ridge-regularized full quadratic to the n_fit best points told
    so far (centred on the best), minimizes it in a ball of the current radius, and
    adds a small isotropic jitter. Radii cycle through `radii` on failure; a new best
    keeps the radius that found it.
    """

    def __init__(self, space, seed=0, n_fit=250, radii=(0.05, 0.1, 0.2, 0.03, 0.3),
                 jitter=0.2, ridge=1e-4):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.rng = np.random.default_rng(seed)
        self.n_fit, self.radii, self.jitter, self.ridge = n_fit, list(radii), jitter, ridge
        self.X, self.y = [], []
        self.k = 0
        self.pending = []
        self.hist = []

    def _propose(self):
        X, y = np.array(self.X), np.array(self.y)
        i = np.argsort(y)[: self.n_fit]
        best = X[np.argmin(y)]
        Z, ys = X[i] - best, y[i]
        F = _feat(Z)
        A = F.T @ F + self.ridge * len(ys) * np.eye(F.shape[1])
        A[0, 0] -= self.ridge * len(ys)
        c = np.linalg.solve(A, F.T @ ys)
        r = self.radii[self.k % len(self.radii)]
        f = lambda z: float((_feat(z) @ c)[0])
        bestres = None
        for _ in range(8):
            z0 = self.rng.normal(size=len(best))
            z0 *= r * self.rng.uniform(0.2, 1.0) / np.linalg.norm(z0)
            res = minimize(f, z0, method="SLSQP",
                           constraints=[{"type": "ineq", "fun": lambda z: r * r - z @ z}])
            if bestres is None or res.fun < bestres.fun:
                bestres = res
        z = bestres.x + self.rng.normal(size=len(best)) * self.jitter * r / np.sqrt(len(best))
        return np.clip(best + z, self.lo, self.hi)

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.y) < 30:
                p = self.rng.uniform(self.lo, self.hi)
            else:
                p = self._propose()
            out.append(p.tolist())
        self.pending += out
        return out

    def tell(self, X, y, g):
        prev = min(self.y) if self.y else np.inf
        for x, v in zip(X, y):
            v = float(v) if np.isfinite(v) else (max(self.y) if self.y else 1e10)
            self.X.append(list(map(float, x)))
            self.y.append(v)
            if x in self.pending:
                self.pending.remove(x)
        if len(X) <= 2:
            if min(self.y) < prev:
                self.hist.append((len(self.y), min(self.y), self.radii[self.k % len(self.radii)]))
            else:
                self.k += 1

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
