"""Local CMA-ES (pycma) in a whitened frame x = m + A z, for continuing an
already-adapted search with a different population size.

Args:
  m: centre (encoded units).
  A: 5x5 (dim x dim) matrix; x = m + A z. Use A = chol of a learned covariance.
  sigma0: initial step size in z units.
  popsize: CMA population.
  n_rep: replicates of m asked first (noise estimate).
  seed: RNG seed.
"""
import pickle
import numpy as np
import cma


class Optimizer:
    max_ask = None
    batch = False

    def __init__(self, space, m, A, sigma0=1.0, popsize=16, n_rep=0, seed=0):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.m = np.array(m, float)
        self.A = np.array(A, float)
        self.n_rep = n_rep
        self.es = cma.CMAEvolutionStrategy(np.zeros(len(self.m)), sigma0,
                                           {"popsize": popsize, "verbose": -9,
                                            "seed": int(seed) + 1})
        self.queue, self.pop, self.raw, self.vals = [], [], [], {}
        self.outstanding = []  # (kind, idx, x)

    def _x(self, z):
        x = self.m + self.A @ np.asarray(z)
        xc = np.clip(x, self.lo, self.hi)
        return xc, float(np.sum((x - xc) ** 2))

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.n_rep > 0:
                self.n_rep -= 1
                x = [float(v) for v in self.m]
                self.outstanding.append(("rep", -1, x))
                out.append(x)
                continue
            if not self.queue:
                if self.pop:
                    break
                self.raw = self.es.ask()
                self.pop = [self._x(z) for z in self.raw]
                self.vals = {}
                self.queue = list(range(len(self.raw)))
            i = self.queue.pop(0)
            x = [float(v) for v in self.pop[i][0]]
            self.outstanding.append(("pop", i, x))
            out.append(x)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            for k, (kind, i, xo) in enumerate(self.outstanding):
                if np.allclose(xo, x, atol=1e-9):
                    self.outstanding.pop(k)
                    if kind == "pop" and self.pop:
                        self.vals[i] = float(v) + 1e4 * self.pop[i][1]
                    break
        if self.pop and len(self.vals) == len(self.pop):
            self.es.tell(self.raw, [self.vals[i] for i in range(len(self.pop))])
            self.queue, self.pop, self.raw, self.vals = [], [], [], {}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
