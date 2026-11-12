import pickle
import numpy as np
import cma


class Optimizer:
    """IPOP-CMA-ES: each restart starts at x0 with sigma0 and doubles the population.
    Points of a generation are handed out one at a time; tell is buffered per generation.
    Tells for points not asked (warm start) are only recorded."""

    def __init__(self, space, seed=0, x0=None, sigma0=1.0, popsize=16, max_popsize=256,
                 tolx=1e-4, tolfun=1e-6):
        b = np.array(space["bounds"], dtype=float)
        self.lb, self.ub = b[:, 0], b[:, 1]
        self.dim = len(self.lb)
        self.seed = seed
        self.x0 = list(x0) if x0 is not None else list((self.lb + self.ub) / 2)
        self.sigma0 = sigma0
        self.popsize = popsize
        self.max_popsize = max_popsize
        self.tolx, self.tolfun = tolx, tolfun
        self.restarts = 0
        self.gen = []          # current generation points (lists)
        self.gen_y = {}        # index -> y
        self.next_idx = 0      # next point of gen to hand out
        self.outstanding = {}  # tuple(point) -> list of gen indices
        self.best_y, self.best_x = np.inf, None
        self.log = []
        self._new_es()

    def _new_es(self):
        opts = {"seed": self.seed + 1000 * self.restarts + 1, "popsize": self.popsize,
                "bounds": [list(self.lb), list(self.ub)], "tolx": self.tolx,
                "tolfun": self.tolfun, "verbose": -9}
        self.es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, opts)
        self.log.append(("start", self.restarts, self.popsize))

    def _new_gen(self):
        if self.es.stop():
            self.log.append(("stop", self.restarts, dict(self.es.stop()), float(self.es.result.fbest)))
            self.restarts += 1
            self.popsize = min(2 * self.popsize, self.max_popsize)
            self._new_es()
        self.gen = [list(map(float, np.clip(p, self.lb, self.ub))) for p in self.es.ask()]
        self.gen_y = {}
        self.next_idx = 0
        self.outstanding = {}

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.next_idx >= len(self.gen):
                if len(self.gen_y) < len(self.gen):
                    break  # wait for this generation's values
                self._new_gen()
            p = self.gen[self.next_idx]
            self.outstanding.setdefault(tuple(p), []).append(self.next_idx)
            self.next_idx += 1
            out.append(p)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v)
            if v < self.best_y:
                self.best_y, self.best_x = v, list(map(float, x))
            k = tuple(map(float, x))
            if k in self.outstanding and self.outstanding[k]:
                self.gen_y[self.outstanding[k].pop(0)] = v
        if self.gen and len(self.gen_y) == len(self.gen):
            self.es.tell([np.array(p) for p in self.gen], [self.gen_y[i] for i in range(len(self.gen))])
            self.gen, self.gen_y, self.next_idx, self.outstanding = [], {}, 0, {}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
