"""IPOP-CMA-ES (pycma) with sequential ask/tell and pickle-based save/load.

Population is asked as a whole from pycma and handed out one point at a time;
the generation is told once every member has a value. On stop (other than
budget), restart with doubled popsize from the best point seen (or a random
point every other restart), sigma reset to sigma0.
"""
import base64
import pickle

import cma
import numpy as np


class Optimizer:
    def __init__(self, space, seed=0, sigma0=2.5, x0=None, popsize=None,
                 restart_random=False):
        self.lo = [b[0] for b in space["bounds"]]
        self.hi = [b[1] for b in space["bounds"]]
        self.d = len(self.lo)
        self.seed = seed
        self.sigma0 = sigma0
        self.x0 = x0
        self.popsize = popsize or (4 + int(3 * np.log(self.d)))
        self.restart_random = restart_random
        self.rng = np.random.default_rng(seed)
        self.es = None
        self.queue = []      # points of current generation not yet asked
        self.gen = []        # all points of current generation
        self.vals = {}       # index in gen -> value
        self.best_x = None
        self.best_y = None
        self.restarts = 0
        self.history = []    # (x, y) of everything told

    def _new_es(self):
        if self.x0 is not None and self.restarts == 0:
            x0 = list(self.x0)
        elif self.best_x is not None and not (self.restart_random and self.restarts % 2 == 1):
            x0 = list(self.best_x)
        else:
            x0 = (np.array(self.lo) + self.rng.random(self.d) *
                  (np.array(self.hi) - np.array(self.lo))).tolist()
        opts = {"bounds": [self.lo, self.hi], "popsize": self.popsize,
                "seed": int(self.rng.integers(1, 1 << 30)), "verbose": -9,
                "tolfun": 1e-12, "tolx": 1e-10}
        self.es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)

    def _new_gen(self):
        if self.es is None:
            self._new_es()
        if self.es.stop():
            self.restarts += 1
            self.popsize *= 2
            self._new_es()
        pts = [list(map(float, p)) for p in self.es.ask()]
        self.gen = pts
        self.queue = list(range(len(pts)))
        self.vals = {}

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                if self.gen and len(self.vals) < len(self.gen):
                    # generation incomplete but all asked: re-ask an untold member
                    missing = [i for i in range(len(self.gen)) if i not in self.vals]
                    out.append(self.gen[missing[len(out) % len(missing)]])
                    continue
                self._new_gen()
            out.append(self.gen[self.queue.pop(0)])
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v)
            self.history.append((list(map(float, x)), v))
            if self.best_y is None or v < self.best_y:
                self.best_y, self.best_x = v, list(map(float, x))
            for i, p in enumerate(self.gen):
                if i not in self.vals and np.allclose(p, x, rtol=0, atol=1e-6):
                    self.vals[i] = v
                    break
        if self.gen and len(self.vals) == len(self.gen):
            self.es.tell([np.array(p) for p in self.gen],
                         [self.vals[i] for i in range(len(self.gen))])
            self.gen, self.queue, self.vals = [], [], {}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
