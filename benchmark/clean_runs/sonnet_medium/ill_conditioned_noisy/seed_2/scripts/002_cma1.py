import pickle, numpy as np, cma

class Optimizer:
    def __init__(self, space, x0=None, sigma=1.5, popsize=12, seed=0, **kw):
        self.lo = np.array([b[0] for b in space["bounds"]]); self.hi = np.array([b[1] for b in space["bounds"]])
        x0 = np.zeros(len(self.lo)) if x0 is None else np.array(x0, float)
        self.es = cma.CMAEvolutionStrategy(list(x0), sigma, {"popsize": popsize, "seed": seed + 1,
            "bounds": [list(self.lo), list(self.hi)], "verbose": -9})
        self.queue = []; self.out = []; self.res = []; self.gen = []
    def ask(self, n):
        pts = []
        for _ in range(n):
            if not self.queue:
                self.gen = [np.array(s) for s in self.es.ask()]; self.queue = list(self.gen); self.res = []
            p = self.queue.pop(0); pts.append(np.clip(p, self.lo, self.hi).tolist())
        return pts
    def tell(self, X, y, g):
        for x, v in zip(X, y):
            self.res.append((np.array(x), float(v)))
        if self.gen and len(self.res) >= len(self.gen) and not self.queue:
            # match by order of ask
            xs = [r[0] for r in self.res[-len(self.gen):]]; ys = [r[1] for r in self.res[-len(self.gen):]]
            self.es.tell(self.gen, ys); self.gen = []; self.res = []
    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self.__dict__, f)
    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f: o.__dict__.update(pickle.load(f))
        return o
