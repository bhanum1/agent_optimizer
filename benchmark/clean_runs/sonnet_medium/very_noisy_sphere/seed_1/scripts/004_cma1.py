import numpy as np, pickle, cma

class Optimizer:
    max_ask = None
    batch = False
    def __init__(self, space, seed=0, sigma=0.5, popsize=10, x0=None, top=1, **kw):
        self.space = space
        lo = np.array([b[0] for b in space["bounds"]]); hi = np.array([b[1] for b in space["bounds"]])
        self.lo, self.hi = lo, hi
        self.args = dict(seed=seed, sigma=sigma, popsize=popsize, x0=x0)
        self.x0 = None if x0 is None else list(x0)
        self.sigma, self.popsize, self.seed = sigma, popsize, seed
        self.es = None
        self.queue = []; self.pending = []; self.vals = []
        self.told = []  # (x,y) history until es exists
    def _make(self):
        x0 = np.array(self.x0 if self.x0 is not None else (self.lo + self.hi) / 2)
        self.es = cma.CMAEvolutionStrategy(list(x0), self.sigma, {
            "bounds": [list(self.lo), list(self.hi)], "popsize": self.popsize,
            "seed": self.seed + 1, "verbose": -9})
    def _gen(self):
        if self.es is None: self._make()
        self.cur = [np.clip(np.array(p), self.lo, self.hi) for p in self.es.ask()]
        self.queue = list(range(len(self.cur))); self.vals = [None] * len(self.cur)
    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue: 
                if all(v is not None for v in self.vals) and self.vals: self._finish()
                if not self.queue and (not self.vals or all(v is not None for v in self.vals)): self._gen()
                if not self.queue: break
            i = self.queue.pop(0); self.pending.append(i)
            out.append([float(v) for v in self.cur[i]])
        return out
    def _finish(self):
        self.es.tell([list(p) for p in self.cur], [float(v) for v in self.vals]); self.vals = []
    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = np.array(x); hit = None
            for i in self.pending:
                if np.allclose(self.cur[i], x): hit = i; break
            if hit is not None:
                self.pending.remove(hit); self.vals[hit] = float(v)
    def save(self, path):
        pickle.dump(self.__dict__, open(path, "wb"))
    @classmethod
    def load(cls, path):
        o = cls.__new__(cls); o.__dict__.update(pickle.load(open(path, "rb"))); return o
