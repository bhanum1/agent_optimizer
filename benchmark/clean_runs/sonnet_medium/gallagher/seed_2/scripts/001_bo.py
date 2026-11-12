import numpy as np, pickle
from scipy.stats import norm, qmc
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, space, seed=0, n_init=10):
        self.b = np.array(space["bounds"], float)
        self.d = len(self.b)
        self.seed = seed
        self.n_init = n_init
        self.X, self.y = [], []
        self.k = 0
        self.rng = np.random.default_rng(seed)
        self.init = qmc.Sobol(self.d, seed=seed).random(16)

    def _u(self, x):
        return (np.array(x) - self.b[:, 0]) / (self.b[:, 1] - self.b[:, 0])

    def _x(self, u):
        return (self.b[:, 0] + np.clip(u, 0, 1) * (self.b[:, 1] - self.b[:, 0])).tolist()

    def _fit(self, U, y):
        def K(A, B, p):
            ls = np.exp(p[:self.d])
            D = ((A[:, None, :] - B[None]) / ls) ** 2
            return np.exp(p[self.d]) * np.exp(-0.5 * D.sum(-1))

        def nll(p):
            Kx = K(U, U, p) + (np.exp(p[self.d + 1]) + 1e-8) * np.eye(len(U))
            try:
                L = np.linalg.cholesky(Kx)
            except np.linalg.LinAlgError:
                return 1e10
            a = np.linalg.solve(L.T, np.linalg.solve(L, y))
            return 0.5 * y @ a + np.log(np.diag(L)).sum()
        best = None
        for s in range(3):
            p0 = np.r_[np.log(0.3) + 0.5 * self.rng.normal(size=self.d), 0.0, np.log(1e-4)]
            r = minimize(nll, p0, method="L-BFGS-B",
                         bounds=[(-4, 1.5)] * self.d + [(-3, 3), (-12, -1)])
            if best is None or r.fun < best.fun:
                best = r
        p = best.x
        Kx = K(U, U, p) + (np.exp(p[self.d + 1]) + 1e-8) * np.eye(len(U))
        L = np.linalg.cholesky(Kx)
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))

        def pred(V):
            Ks = K(V, U, p)
            m = Ks @ a
            v = np.linalg.solve(L, Ks.T)
            s2 = np.maximum(np.exp(p[self.d]) - (v ** 2).sum(0), 1e-12)
            return m, np.sqrt(s2)
        return pred

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.y) < self.n_init:
                u = self.init[len(self.y) % 16]
            else:
                y = np.array(self.y)
                mu, sd = y.mean(), y.std() + 1e-12
                z = (y - mu) / sd
                pred = self._fit(np.array([self._u(x) for x in self.X]), z)
                fb = z.min()

                def ei(V):
                    m, s = pred(V)
                    t = (fb - m) / s
                    return s * (t * norm.cdf(t) + norm.pdf(t))
                C = self.rng.random((4000, self.d))
                Xu = np.array([self._u(x) for x in self.X])
                C = np.vstack([C, np.clip(Xu[np.argmin(z)] + 0.05 * self.rng.normal(size=(1000, self.d)), 0, 1)])
                e = ei(C)
                u = C[np.argsort(-e)[0]]
                r = minimize(lambda v: -ei(v[None])[0], u, method="L-BFGS-B", bounds=[(0, 1)] * self.d)
                if -r.fun >= e.max():
                    u = r.x
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
