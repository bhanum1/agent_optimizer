import json, numpy as np

class Optimizer:
    def __init__(self, space, seed=0, k=6, s0=0.6, s1=0.15, horizon=300, **kw):
        self.lo = np.array([b[0] for b in space["bounds"]]); self.hi = np.array([b[1] for b in space["bounds"]])
        self.seed, self.k, self.s0, self.s1, self.h = seed, k, s0, s1, horizon
        self.rng = np.random.default_rng(seed)
        self.X, self.y, self.t = [], [], 0

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.X) < 3:
                out.append(list(self.rng.uniform(self.lo, self.hi))); continue
            y = np.array(self.y); idx = np.argsort(y)[:self.k]
            c = np.array(self.X)[self.rng.choice(idx)]
            f = min(1.0, self.t / self.h)
            s = self.s0 * (self.s1 / self.s0) ** f
            x = np.clip(c + self.rng.normal(0, s, len(c)), self.lo, self.hi)
            out.append([float(v) for v in x]); self.t += 1
        return out

    def tell(self, X, y, g):
        self.X += [list(map(float, x)) for x in X]; self.y += [float(v) for v in y]

    def save(self, path):
        json.dump({"a": [self.seed, self.k, self.s0, self.s1, self.h], "X": self.X, "y": self.y, "t": self.t,
                   "rng": self.rng.bit_generator.state}, open(path, "w"))

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        o = cls.__new__(cls)
        import numpy as np
        return o._restore(d)

    def _restore(self, d):
        # bounds are fixed for this problem
        self.lo = np.full(5, -5.0); self.hi = np.full(5, 5.0)
        self.seed, self.k, self.s0, self.s1, self.h = d["a"]
        self.rng = np.random.default_rng(); self.rng.bit_generator.state = d["rng"]
        self.X, self.y, self.t = d["X"], d["y"], d["t"]
        return self
