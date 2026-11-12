import base64
import pickle

import cma
import numpy as np


class Optimizer:
    """IPOP-CMA-ES in the unit cube, asked one point at a time (tolx and a per-run
    evaluation cap settable; run starts logged in run_starts).

    The first run starts at the best told point (or the cube centre). When a run
    stops, the next one starts at the best point so far (or, every `random_every`-th
    restart, a uniform random point) with the population multiplied by `inc_popsize`.
    """

    def __init__(self, space, seed=0, sigma0=0.1, popsize=None, inc_popsize=2.0,
                 random_every=0, tolflatfitness=20, max_popsize=256, tolx=1e-6,
                 max_run_evals=0):
        self.lb = np.array(space["bounds"], dtype=float)[:, 0]
        self.ub = np.array(space["bounds"], dtype=float)[:, 1]
        self.d = len(self.lb)
        self.seed = int(seed)
        self.sigma0 = float(sigma0)
        self.popsize0 = int(popsize) if popsize else 4 + int(3 * np.log(self.d))
        self.popsize = self.popsize0
        self.inc_popsize = float(inc_popsize)
        self.random_every = int(random_every)
        self.tolflatfitness = int(tolflatfitness)
        self.max_popsize = int(max_popsize)
        self.tolx = float(tolx)
        self.max_run_evals = int(max_run_evals)  # 0: no cap on evaluations per run
        self.rng = np.random.default_rng(self.seed)
        self.best_u, self.best_y = None, np.inf
        self.n_runs = 0
        self.run_starts = []  # (eval count, popsize, start point) per run
        self.n_told = 0
        self.es = None
        self.pop = []       # current population (unit points)
        self.vals = []      # value per population member, None until told
        self.next_i = 0     # next population member to hand out

    def _to_unit(self, x):
        return (np.asarray(x, float) - self.lb) / (self.ub - self.lb)

    def _from_unit(self, u):
        return (self.lb + np.asarray(u, float) * (self.ub - self.lb)).tolist()

    def _new_run(self):
        if self.n_runs > 0:
            self.popsize = min(int(round(self.popsize * self.inc_popsize)), self.max_popsize)
        rand = self.random_every > 0 and self.n_runs > 0 and self.n_runs % self.random_every == 0
        if self.best_u is None or rand:
            x0 = self.rng.uniform(0, 1, self.d) if self.best_u is not None else np.full(self.d, 0.5)
        else:
            x0 = np.array(self.best_u)
        opts = {"bounds": [0, 1], "popsize": self.popsize, "verbose": -9,
                "seed": int(self.rng.integers(1, 2**31)), "tolflatfitness": self.tolflatfitness,
                "tolfun": 1e-9, "tolx": self.tolx}
        self.es = cma.CMAEvolutionStrategy(np.clip(x0, 1e-6, 1 - 1e-6).tolist(), self.sigma0, opts)
        self.n_runs += 1
        self.run_starts.append((self.n_told, self.popsize, [float(v) for v in x0]))
        self._new_pop()

    def _new_pop(self):
        self.pop = [np.clip(np.asarray(p, float), 0, 1).tolist() for p in self.es.ask()]
        self.vals = [None] * len(self.pop)
        self.next_i = 0

    def ask(self, n):
        if self.es is None:
            self._new_run()
        out = []
        while len(out) < n and self.next_i < len(self.pop):
            out.append(self._from_unit(self.pop[self.next_i]))
            self.next_i += 1
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            self.n_told += 1
            u = self._to_unit(x)
            v = float(v) if np.isfinite(v) else 1e12
            if v < self.best_y:
                self.best_y, self.best_u = v, u.tolist()
            for i, p in enumerate(self.pop):
                if self.vals[i] is None and i < self.next_i and np.allclose(u, p, atol=1e-9):
                    self.vals[i] = v
                    break
        if self.pop and all(v is not None for v in self.vals):
            self.es.tell([np.array(p) for p in self.pop], self.vals)
            capped = self.max_run_evals > 0 and self.es.countevals >= self.max_run_evals
            if self.es.stop() or capped:
                self._new_run()
            else:
                self._new_pop()

    def save(self, path):
        d = dict(self.__dict__)
        d["rng"] = self.rng.bit_generator.state
        with open(path, "wb") as f:
            pickle.dump(d, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        o = cls.__new__(cls)
        rs = d.pop("rng")
        o.__dict__.update(d)
        o.rng = np.random.default_rng()
        o.rng.bit_generator.state = rs
        return o
