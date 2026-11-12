import pickle
import numpy as np
from scipy.stats import norm, qmc
from scipy.optimize import minimize


class Optimizer:
    max_ask = 1
    batch = False

    def __init__(self, space, seed=0, n_init=10):
        self.b = np.array(space["bounds"], float)
        self.d = len(self.b)
        self.seed = seed
        self.n_init = n_init
        self.rng = np.random.default_rng(seed)
        self.sobol = qmc.Sobol(self.d, seed=seed)
        self.X, self.y = [], []
        self.pending = []

    def _u(self, x):
        return (np.asarray(x) - self.b[:, 0]) / (self.b[:, 1] - self.b[:, 0])

    def _x(self, u):
        return (self.b[:, 0] + np.asarray(u) * (self.b[:, 1] - self.b[:, 0])).tolist()

    def _fit(self, U, y):
        mu, sd = y.mean(), y.std() + 1e-12
        z = (y - mu) / sd

        def K(A, B, ls):
            d2 = (((A[:, None, :] - B[None, :, :]) / ls) ** 2).sum(-1)
            return np.exp(-0.5 * d2)

        def nll(p):
            ls = np.exp(p[:self.d]); nz = np.exp(p[self.d]) + 1e-8
            Kx = K(U, U, ls) + nz * np.eye(len(U))
            try:
                L = np.linalg.cholesky(Kx)
            except np.linalg.LinAlgError:
                return 1e10
            a = np.linalg.solve(L.T, np.linalg.solve(L, z))
            return 0.5 * z @ a + np.log(np.diag(L)).sum()

        best = None
        for p0 in ([np.log(0.3)] * self.d + [np.log(1e-4)], [np.log(0.15)] * self.d + [np.log(1e-6)]):
            r = minimize(nll, p0, method="L-BFGS-B",
                         bounds=[(np.log(0.03), np.log(3))] * self.d + [(np.log(1e-8), np.log(0.1))])
            if best is None or r.fun < best.fun:
                best = r
        ls = np.exp(best.x[:self.d]); nz = np.exp(best.x[self.d]) + 1e-8
        L = np.linalg.cholesky(K(U, U, ls) + nz * np.eye(len(U)))
        a = np.linalg.solve(L.T, np.linalg.solve(L, z))

        def pred(Q):
            Ks = K(Q, U, ls)
            m = Ks @ a
            v = np.linalg.solve(L, Ks.T)
            s2 = np.maximum(1 - (v ** 2).sum(0), 1e-12)
            return m, np.sqrt(s2)
        return pred, (z.min())

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.y) + len(out) < self.n_init and len(self.y) < self.n_init:
                k = len(self.y) + len(out)
                if k < self.n_init:
                    u = self.sobol.random(1)[0]
                    out.append(self._x(u)); continue
            U = np.array([self._u(x) for x in self.X])
            y = np.array(self.y)
            pred, zmin = self._fit(U, y)
            C = self.rng.random((4000, self.d))
            # local candidates around best points
            order = np.argsort(y)[:3]
            for i in order:
                C = np.vstack([C, np.clip(U[i] + 0.05 * self.rng.standard_normal((500, self.d)), 0, 1)])
            m, s = pred(C)
            imp = zmin - m
            ei = imp * norm.cdf(imp / s) + s * norm.pdf(imp / s)
            u = C[np.argmax(ei)]
            out.append(self._x(u))
        self.pending = out
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            self.X.append(list(x)); self.y.append(float(v))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
