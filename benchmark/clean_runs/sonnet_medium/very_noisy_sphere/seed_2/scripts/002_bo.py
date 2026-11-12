import numpy as np, pickle
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

class Optimizer:
    def __init__(self, space, seed=0, n_init=0, refit=8, ncand=3000):
        self.b = np.array(space["bounds"], float); self.d = len(self.b)
        self.rng = np.random.default_rng(seed)
        self.X = []; self.y = []; self.pending = []
        self.refit = refit; self.ncand = ncand; self.theta = None; self.since = 999
    def _u(self, X): return (np.array(X, float) - self.b[:, 0]) / (self.b[:, 1] - self.b[:, 0])
    def _k(self, A, B, th):
        ls = np.exp(th[:self.d]); A = A / ls; B = B / ls
        d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T
        r = np.sqrt(np.maximum(d2, 0)) * np.sqrt(5)
        return np.exp(th[self.d]) * (1 + r + r**2 / 3) * np.exp(-r)
    def _nll(self, th, U, z):
        K = self._k(U, U, th) + (np.exp(th[self.d + 1]) + 1e-6) * np.eye(len(U))
        try: c = cho_factor(K)
        except Exception: return 1e10
        a = cho_solve(c, z)
        return 0.5 * z @ a + np.log(np.diag(c[0])).sum() + 0.01 * (np.sum((th[:self.d] + 1.0) ** 2))
    def _fit(self):
        U = self._u(self.X); y = np.array(self.y)
        self.mu, self.sd = y.mean(), y.std() + 1e-9
        z = (y - self.mu) / self.sd
        if self.theta is None or self.since >= self.refit:
            x0 = self.theta if self.theta is not None else np.r_[np.full(self.d, np.log(0.4)), 0.0, np.log(0.05)]
            bd = [(np.log(0.05), np.log(5))] * self.d + [(np.log(0.05), np.log(20)), (np.log(1e-4), np.log(2))]
            self.theta = minimize(self._nll, x0, args=(U, z), bounds=bd, method="L-BFGS-B").x
            self.since = 0
        th = self.theta
        K = self._k(U, U, th) + (np.exp(th[self.d + 1]) + 1e-6) * np.eye(len(U))
        self.c = cho_factor(K); self.alpha = cho_solve(self.c, z); self.U = U; self.z = z
    def _pred(self, C):
        Ks = self._k(C, self.U, self.theta)
        m = Ks @ self.alpha
        v = np.exp(self.theta[self.d]) - (Ks * cho_solve(self.c, Ks.T).T).sum(1)
        return m, np.sqrt(np.maximum(v, 1e-12))
    def ask(self, n):
        out = []
        if len(self.X) < 5:
            return [list(self.b[:, 0] + self.rng.random(self.d) * (self.b[:, 1] - self.b[:, 0])) for _ in range(n)]
        self._fit()
        m, _ = self._pred(self.U); best = np.argmin(m)
        ub = self.U[best]
        for _ in range(n):
            top = np.argsort(m)[:5]
            parts = [self.rng.random((self.ncand // 3, self.d))]
            for s, f in ((0.03, 1), (0.1, 1)):
                cen = self.U[self.rng.choice(top, self.ncand // 3)]
                parts.append(np.clip(cen + self.rng.normal(0, s, cen.shape), 0, 1))
            C = np.vstack(parts)
            mc, sc = self._pred(C)
            # EI on GP mean (noise-robust): incumbent = best posterior mean
            inc = m.min(); g = (inc - mc) / sc
            ei = sc * (g * norm.cdf(g) + norm.pdf(g))
            x = C[np.argmax(ei)]
            out.append(list(self.b[:, 0] + x * (self.b[:, 1] - self.b[:, 0])))
        self.pending = out
        return out
    def tell(self, X, y, g):
        self.X += [list(map(float, x)) for x in X]; self.y += [float(v) for v in y]; self.since += len(y)
    def save(self, path): pickle.dump(self.__dict__, open(path, "wb"))
    @classmethod
    def load(cls, path):
        o = cls.__new__(cls); o.__dict__.update(pickle.load(open(path, "rb"))); return o
