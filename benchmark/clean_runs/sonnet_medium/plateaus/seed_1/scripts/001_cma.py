import pickle
import numpy as np
import cma
from scipy.stats import qmc


class Optimizer:
    """Sobol initial design, then CMA-ES with restarts (popsize doubles each restart)."""
    max_ask = 1

    def __init__(self, space, seed=0, n_init=64, sigma0=0.3, popsize=None):
        self.b = np.array(space["bounds"], float)
        self.d = len(self.b)
        self.seed = seed
        self.sigma0 = sigma0
        self.pop0 = popsize or 4 + int(3 * np.log(self.d))
        self.pop = self.pop0
        self.restarts = 0
        self.init = qmc.Sobol(self.d, seed=seed).random(n_init).tolist()
        self.best = (np.inf, None)
        self.es = None
        self.queue = []      # unit-cube points awaiting ask
        self.pending = []    # (unit point) asked, not told
        self.cur = []        # unit points of current generation
        self.res = []        # values for current generation
        self.n_init_left = n_init

    def _new_es(self):
        rs = self.restarts
        if rs == 0 and self.best[1] is not None:
            x0 = self.best[1]
            sig = self.sigma0
        else:
            rng = np.random.default_rng(self.seed * 1000 + rs)
            x0 = rng.uniform(0.1, 0.9, self.d)
            sig = 0.3
            self.pop *= 2
        self.es = cma.CMAEvolutionStrategy(
            list(x0), sig,
            {"bounds": [0, 1], "popsize": self.pop, "seed": self.seed + rs + 1, "verbose": -9})
        self.restarts += 1

    def _gen(self):
        if self.es is None or self.es.stop():
            if self.es is not None and self.es.stop():
                pass
            self._new_es()
        self.cur = [list(map(float, x)) for x in self.es.ask()]
        self.res = []
        self.queue = list(self.cur)

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.init:
                u = self.init.pop(0)
            else:
                if not self.queue:
                    self._gen()
                u = self.queue.pop(0)
            self.pending.append(u)
            out.append((self.b[:, 0] + np.array(u) * (self.b[:, 1] - self.b[:, 0])).tolist())
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            u = (np.array(x) - self.b[:, 0]) / (self.b[:, 1] - self.b[:, 0])
            if v < self.best[0]:
                self.best = (float(v), u.copy())
            match = None
            for i, p in enumerate(self.pending):
                if np.allclose(p, u, atol=1e-9):
                    match = i
                    break
            if match is None:
                continue
            p = self.pending.pop(match)
            if self.es is not None and any(np.allclose(p, c, atol=1e-12) for c in self.cur) and len(self.res) < len(self.cur):
                self.res.append((p, float(v)))
                if len(self.res) == len(self.cur):
                    self.es.tell([r[0] for r in self.res], [r[1] for r in self.res])
                    self.cur = []
                    self.res = []

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
