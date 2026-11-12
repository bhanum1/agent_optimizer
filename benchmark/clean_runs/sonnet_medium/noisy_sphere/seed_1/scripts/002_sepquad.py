import json, numpy as np

class Optimizer:
    def __init__(self, space, seed=0, sig0=0.6, sig1=0.15, n_explore=20, lam=1e-3):
        self.lo = np.array([b[0] for b in space["bounds"]]); self.hi = np.array([b[1] for b in space["bounds"]])
        self.d = space["dim"]; self.sp_bounds = space["bounds"]; self.seed = seed; self.sig0 = sig0; self.sig1 = sig1
        self.n_explore = n_explore; self.lam = lam
        self.X = []; self.y = []; self.nask = 0
        self.rng = np.random.default_rng(seed)

    def _fit(self):
        X = np.array(self.X); y = np.array(self.y); d = self.d
        A = np.hstack([np.ones((len(X), 1)), X, X ** 2])
        P = np.eye(A.shape[1]) * self.lam; P[0, 0] = 0
        w = np.linalg.solve(A.T @ A + P, A.T @ y)
        b = w[1:d + 1]; a = np.maximum(w[d + 1:], 1e-3)
        return np.clip(-b / (2 * a), self.lo, self.hi)

    def ask(self, n):
        out = []
        for _ in range(n):
            m = self._fit() if len(self.X) > 2 * self.d + 2 else (self.lo + self.hi) / 2
            sig = self.sig0 if self.nask < self.n_explore else self.sig1
            x = np.clip(m + sig * self.rng.standard_normal(self.d), self.lo, self.hi)
            self.nask += 1
            out.append([float(v) for v in x])
        return out

    def tell(self, X, y, g):
        self.X += [list(map(float, x)) for x in X]; self.y += [float(v) for v in y]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"X": self.X, "y": self.y, "nask": self.nask, "args": [self.seed, self.sig0, self.sig1, self.n_explore, self.lam],
                       "rng": self.rng.bit_generator.state, "space": {"bounds": self.sp_bounds, "dim": self.d}}, f)

    @classmethod
    def load(cls, path):
        s = json.load(open(path))
        o = cls.__new__(cls)
        return o._restore(s)

    def _restore(self, s):
        self.__init__(s["space"], *s["args"])
        self.X = s["X"]; self.y = s["y"]; self.nask = s["nask"]
        self.rng.bit_generator.state = s["rng"]
        return self
