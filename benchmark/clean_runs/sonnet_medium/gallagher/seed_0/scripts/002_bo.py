import pickle
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm


def _k(A, B, ls, var):
    d = (A[:, None, :] - B[None, :, :]) / ls
    r = np.sqrt((d ** 2).sum(-1)) * np.sqrt(5)
    return var * (1 + r + r ** 2 / 3) * np.exp(-r)


class Optimizer:
    def __init__(self, space, seed=0, n_init=0, xi=0.0):
        self.space = space
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.rng = np.random.default_rng(seed)
        self.xi = xi
        self.X, self.y = [], []

    def _u(self, x):
        return (np.asarray(x, float) - self.lo) / (self.hi - self.lo)

    def _fit(self):
        U = np.array([self._u(x) for x in self.X])
        y = np.array(self.y)
        mu, sd = y.mean(), y.std() + 1e-9
        z = (y - mu) / sd
        best = None
        for ls0 in (0.1, 0.2, 0.4):
            def nll(p):
                ls, var = np.exp(p[0]), np.exp(p[1])
                K = _k(U, U, ls, var) + 1e-6 * np.eye(len(U))
                try:
                    c = cho_factor(K)
                except Exception:
                    return 1e10
                a = cho_solve(c, z)
                return 0.5 * z @ a + np.log(np.diag(c[0])).sum()
            r = minimize(nll, [np.log(ls0), 0.0], bounds=[(np.log(0.03), np.log(2)), (np.log(0.1), np.log(10))])
            if best is None or r.fun < best.fun:
                best = r
        ls, var = np.exp(best.x)
        K = _k(U, U, ls, var) + 1e-6 * np.eye(len(U))
        c = cho_factor(K)
        return U, z, mu, sd, ls, var, c, cho_solve(c, z)

    def ask(self, n):
        if len(self.X) < 3:
            return [list(self.lo + self.rng.random(len(self.lo)) * (self.hi - self.lo)) for _ in range(n)]
        U, z, mu, sd, ls, var, c, a = self._fit()
        fb = z.min()

        def ei(u):
            u = np.atleast_2d(u)
            ks = _k(u, U, ls, var)
            m = ks @ a
            v = np.maximum(var - np.einsum("ij,ji->i", ks, cho_solve(c, ks.T)), 1e-12)
            s = np.sqrt(v)
            g = (fb - m - self.xi) / s
            return s * (g * norm.cdf(g) + norm.pdf(g))

        cand = self.rng.random((4000, len(self.lo)))
        cand = np.vstack([cand, np.clip(U[np.argsort(z)[:3]][:, None, :] + 0.03 * self.rng.standard_normal((3, 300, len(self.lo))), 0, 1).reshape(-1, len(self.lo))])
        e = ei(cand)
        bestu, beste = cand[e.argmax()], e.max()
        for i in np.argsort(-e)[:5]:
            r = minimize(lambda u: -ei(u)[0], cand[i], bounds=[(0, 1)] * len(self.lo))
            if -r.fun > beste:
                bestu, beste = r.x, -r.fun
        x = self.lo + bestu * (self.hi - self.lo)
        return [list(map(float, x))]

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            self.X.append(list(map(float, x)))
            self.y.append(float(v))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
