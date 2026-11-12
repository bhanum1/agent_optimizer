import numpy as np, pickle, cma

class Optimizer:
    def __init__(self, space, seed=0, x0=None, sigma=1.5, popsize=None):
        self.b = np.array(space["bounds"], float)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.sigma = sigma
        self.pop0 = popsize
        self.pop = popsize
        self.restarts = 0
        self.x0 = list(x0) if x0 is not None else self.rng.uniform(self.b[:, 0], self.b[:, 1]).tolist()
        self.queue = []      # asked from es, not yet handed out
        self.out = []        # handed out, awaiting tell
        self.done = []       # (x, y) for current generation
        self.cur = 0
        self._new_es()

    def _new_es(self):
        o = {"bounds": [self.b[:, 0].tolist(), self.b[:, 1].tolist()], "seed": int(self.rng.integers(1, 2**30)), "verbose": -9}
        if self.pop:
            o["popsize"] = int(self.pop)
        self.es = cma.CMAEvolutionStrategy(self.x0, self.sigma, o)
        self.queue, self.out, self.done = [], [], []

    def ask(self, n):
        pts = []
        for _ in range(n):
            if not self.queue:
                self.queue = [list(map(float, s)) for s in self.es.ask()]
                self.gen_n = len(self.queue)
                self.done = []
            p = self.queue.pop(0)
            self.out.append(p)
            pts.append(p)
        return pts

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = list(map(float, x))
            if x in self.out:
                self.out.remove(x)
                self.done.append((x, float(v)))
        if self.done and not self.queue and not self.out and len(self.done) >= self.gen_n:
            xs, fs = zip(*self.done)
            self.es.tell(list(xs), list(fs))
            self.done = []
            if self.es.stop():
                self.restarts += 1
                best = self.es.result.xbest
                self.pop = int((self.pop or self.es.popsize) * 2) if self.restarts % 2 else self.pop0
                self.x0 = self.rng.uniform(self.b[:, 0], self.b[:, 1]).tolist()
                self.sigma = 2.0
                self._new_es()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
