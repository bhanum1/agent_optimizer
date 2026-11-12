import pickle, numpy as np, cma


class Optimizer:
    max_ask = 1
    batch = False

    def __init__(self, space, seed=0, sigma=2.0, popsize=16, x0=None):
        self.space = space
        lo = [b[0] for b in space["bounds"]]
        hi = [b[1] for b in space["bounds"]]
        x0 = list(x0) if x0 is not None else [0.0] * space["dim"]
        self.es = cma.CMAEvolutionStrategy(
            x0, sigma, {"bounds": [lo, hi], "popsize": popsize, "seed": seed + 1, "verbose": -9})
        self.queue = []   # points of the current population not yet handed out
        self.pop = []     # current population
        self.vals = {}    # index in pop -> y
        self.out = []     # indices handed out, awaiting tell

    def _fill(self):
        self.pop = [list(map(float, p)) for p in self.es.ask()]
        self.queue = list(range(len(self.pop)))
        self.vals = {}

    def ask(self, n):
        pts = []
        for _ in range(n):
            if not self.queue:
                self._fill()
            i = self.queue.pop(0)
            self.out.append(i)
            pts.append(self.pop[i])
        return pts

    def tell(self, X, y, g):
        for k, v in enumerate(y):
            if self.out:
                i = self.out.pop(0)
                self.vals[i] = float(v)
            # extra (warm-start) points are ignored
        if self.pop and len(self.vals) == len(self.pop):
            self.es.tell(self.pop, [self.vals[i] for i in range(len(self.pop))])
            self.pop, self.vals, self.queue = [], {}, []

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
