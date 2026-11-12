import json
import numpy as np


class Optimizer:
    """Sequential local quadratic model: refit on the most recent n_fit points,
    step to the (trust-limited) stationary point; every other ask adds a small
    perturbation along the Hessian eigenvectors to keep the fit well posed."""

    def __init__(self, space, n_fit=60, trust=2.0, jitter=0.5, seed=0):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.n_fit, self.trust, self.jitter = n_fit, trust, jitter
        self.rng = np.random.default_rng(seed)
        self.X, self.y, self.k = [], [], 0

    def _fit(self):
        X = np.array(self.X[-self.n_fit:]); y = np.array(self.y[-self.n_fit:])
        d = X.shape[1]; c = X.mean(0); Z = X - c
        cols = [np.ones(len(Z))] + [Z[:, i] for i in range(d)] + \
               [Z[:, i] * Z[:, j] for i in range(d) for j in range(i, d)]
        coef = np.linalg.lstsq(np.array(cols).T, y, rcond=None)[0]
        b = coef[1:d + 1]; H = np.zeros((d, d)); k = d + 1
        for i in range(d):
            for j in range(i, d):
                if i == j: H[i, i] = 2 * coef[k]
                else: H[i, j] = H[j, i] = coef[k]
                k += 1
        return c, Z, b, H

    def ask(self, n):
        out = []
        for _ in range(n):
            best = np.array(self.X[int(np.argmin(self.y))])
            if len(self.X) < 21 + 5:
                out.append(best.tolist()); continue
            c, Z, b, H = self._fit()
            w, V = np.linalg.eigh(H)
            sd = np.maximum((Z @ V).std(0), 1e-7)
            g = b + H @ (best - c)                    # model gradient at best
            gv = V.T @ g
            # Newton step in eigen coords; skip non-positive curvature directions
            step = np.where(w > 0, -gv / np.where(w > 0, w, 1), 0.0)
            step = np.clip(step, -self.trust * sd, self.trust * sd)
            if self.k % 2 == 1:
                step = step + self.jitter * sd * self.rng.standard_normal(len(sd))
            x = np.clip(best + V @ step, self.lo, self.hi)
            out.append(x.tolist()); self.k += 1
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append([float(v) for v in xi]); self.y.append(float(yi))

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"lo": self.lo.tolist(), "hi": self.hi.tolist(), "n_fit": self.n_fit,
                       "trust": self.trust, "jitter": self.jitter, "X": self.X, "y": self.y,
                       "k": self.k, "rng": self.rng.bit_generator.state}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            s = json.load(f)
        o = cls.__new__(cls)
        o.lo, o.hi = np.array(s["lo"]), np.array(s["hi"])
        o.n_fit, o.trust, o.jitter = s["n_fit"], s["trust"], s["jitter"]
        o.X, o.y, o.k = s["X"], s["y"], s["k"]
        o.rng = np.random.default_rng(); o.rng.bit_generator.state = s["rng"]
        return o
