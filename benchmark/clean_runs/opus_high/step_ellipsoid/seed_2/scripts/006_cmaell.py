import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES in coordinates whitened by the elite points' covariance.

    Each run starts at the best point told so far, in z-coordinates where
    x = best + L z and L L^T = cov(best `n_elite` points) (eigen sds floored at
    `sd_floor`). Runs restart after `stall` evaluations without improvement
    (or cma's own stop tests); the sigma of successive runs cycles through `sigmas`.
    """

    def __init__(self, space, seed=0, n_elite=40, sd_floor=0.02, sigmas=(1.0, 0.5, 2.0),
                 popsize=12, stall=60, tolflat=8):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.d = len(self.lo)
        self.rng = np.random.default_rng(seed)
        self.n_elite, self.sd_floor = n_elite, sd_floor
        self.sigmas = list(sigmas)
        self.popsize, self.stall, self.tolflat = popsize, stall, tolflat
        self.X, self.y = [], []
        self.best_x, self.best_y = None, np.inf
        self.k = 0
        self.es = None
        self.c, self.L = None, None
        self.pop, self.vals, self.issued = [], {}, 0
        self.since, self.run_best = 0, np.inf
        self.hist = []

    def _frame(self):
        X, y = np.array(self.X), np.array(self.y)
        E = X[np.argsort(y)[:self.n_elite]]
        w, V = np.linalg.eigh(np.cov(E.T))
        sd = np.maximum(np.sqrt(np.maximum(w, 0)), self.sd_floor)
        self.c, self.L = self.best_x.copy(), V * sd

    def _to_x(self, z):
        return np.clip(self.c + self.L @ np.asarray(z), self.lo, self.hi)

    def _new_es(self):
        self._frame()
        sig = self.sigmas[self.k % len(self.sigmas)]
        opts = {"popsize": int(self.popsize), "verbose": -9,
                "seed": int(self.rng.integers(1, 2**31 - 1)),
                "tolflatfitness": self.tolflat, "tolx": 1e-8}
        self.es = cma.CMAEvolutionStrategy([0.0] * self.d, sig, opts)
        self.run_best, self.since = np.inf, 0

    def _next_gen(self):
        if self.es is None or self.es.stop() or self.since >= self.stall:
            if self.es is not None:
                self.hist.append((self.k, float(self.run_best)))
                self.k += 1
            self._new_es()
        self.zpop = [np.array(z) for z in self.es.ask()]
        self.pop = [self._to_x(z) for z in self.zpop]
        self.vals, self.issued = {}, 0

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.issued >= len(self.pop):
                if len(self.vals) < len(self.pop):
                    break
                self._next_gen()
            out.append([float(v) for v in self.pop[self.issued]])
            self.issued += 1
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x, v = np.array(x, float), float(v)
            self.X.append(x)
            self.y.append(v)
            if v < self.best_y:
                self.best_y, self.best_x = v, x.copy()
            for i in range(self.issued):
                if i not in self.vals and np.allclose(self.pop[i], x, atol=1e-12):
                    self.vals[i] = v
                    if v < self.run_best - 1e-9:
                        self.run_best, self.since = v, 0
                    else:
                        self.since += 1
                    break
        if self.pop and len(self.vals) == len(self.pop):
            self.es.tell([list(z) for z in self.zpop],
                         [self.vals[i] for i in range(len(self.pop))])
            self.pop, self.vals, self.issued = [], {}, 0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
