"""CMA-ES with IPOP-style restarts (minimization, continuous box).

Each run asks a full population, hands it out one point at a time, and tells CMA when
every point of the generation is back. When a run stops (cma's own criteria, or tolx /
stagnation), a new run starts with a doubled population, alternating between the
global best point (sigma0) and a uniform random point (sigma_global).
Rank-based selection makes it insensitive to plateaus and flat valley floors.
"""
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import cma


class Optimizer:
    max_ask = 1
    batch = False

    def __init__(self, space, seed=0, x0=None, sigma0=0.5, sigma_global=2.5,
                 popsize=None, max_popsize=64, tolx=1e-6, alternate=True):
        self.lb = np.array([b[0] for b in space["bounds"]], float)
        self.ub = np.array([b[1] for b in space["bounds"]], float)
        self.dim = len(self.lb)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sigma0 = sigma0
        self.sigma_global = sigma_global
        self.base_pop = popsize or (4 + int(3 * np.log(self.dim)))
        self.max_popsize = max_popsize
        self.tolx = tolx
        self.alternate = alternate
        self.best_x = None if x0 is None else np.array(x0, float)
        self.best_y = np.inf
        self.n_runs = 0
        self.es = None
        self.queue = []          # points of the current generation not yet asked
        self.gen_X = []          # the current generation, in ask order
        self.gen_y = {}          # index in generation -> value
        self.pending = {}        # tuple(point) -> index in generation
        self.log = []

    def _start_run(self):
        k = self.n_runs
        pop = min(self.base_pop * 2 ** (k // 2 if self.alternate else k), self.max_popsize)
        if self.best_x is not None and (k % 2 == 0 or not self.alternate):
            x0, s0 = self.best_x.copy(), self.sigma0
        else:
            x0, s0 = self.rng.uniform(self.lb, self.ub), self.sigma_global
        opts = {"bounds": [list(self.lb), list(self.ub)], "popsize": pop,
                "seed": int(self.rng.integers(1, 2 ** 31)), "verbose": -9,
                "tolx": self.tolx, "tolfunhist": 0, "tolflatfitness": 50}
        self.es = cma.CMAEvolutionStrategy(list(x0), s0, opts)
        self.n_runs += 1
        self.log.append((self.n_runs, pop, s0))

    def _new_generation(self):
        if self.es is None or self.es.stop():
            self._start_run()
        self.gen_X = [np.clip(np.array(x, float), self.lb, self.ub) for x in self.es.ask()]
        self.gen_y = {}
        self.queue = list(range(len(self.gen_X)))

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                if self.gen_X and len(self.gen_y) < len(self.gen_X):
                    break  # generation still outstanding
                self._new_generation()
            i = self.queue.pop(0)
            x = [float(v) for v in self.gen_X[i]]
            self.pending[tuple(x)] = i
            out.append(x)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = [float(t) for t in x]
            v = float(v)
            if v < self.best_y:
                self.best_y, self.best_x = v, np.array(x)
            i = self.pending.pop(tuple(x), None)
            if i is not None:
                self.gen_y[i] = v
        if self.gen_X and len(self.gen_y) == len(self.gen_X):
            self.es.tell([list(p) for p in self.gen_X],
                         [self.gen_y[i] for i in range(len(self.gen_X))])
            self.gen_X, self.gen_y = [], {}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
