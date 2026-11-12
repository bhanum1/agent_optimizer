import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES with IPOP restarts in whitened coordinates z, x = clip(center + T z).
    Asks one generation at a time, tells it when complete. Restarts are centred on z=0
    or the best z (no random restarts). Points told that were not asked only update
    the incumbent."""

    def __init__(self, space, center, T, x0=None, sigma0=1.0, popsize=None, seed=0,
                 restart_from_best=True, inc_popsize=2):
        self.xlo = np.array(space["bounds"], float)[:, 0]
        self.xhi = np.array(space["bounds"], float)[:, 1]
        self.dim = space["dim"]
        self.center = np.array(center, float)
        self.T = np.array(T, float)
        self.bounds = np.array([[-60.0, 60.0]] * self.dim)
        self.best_z = None
        self.sigma0 = float(sigma0)
        self.seed = int(seed)
        self.inc_popsize = inc_popsize
        self.restart_from_best = restart_from_best
        self.rng = np.random.default_rng(seed)
        self.x0 = None if x0 is None else list(map(float, x0))
        self.popsize = popsize
        self.best_x, self.best_y = None, np.inf
        self.n_restarts = 0
        self.es = None
        self.queue = []      # asked-for-generation points not yet handed out
        self.gen_X = []      # current generation points
        self.gen_y = {}      # index -> value
        self.handed = []     # indices handed out, awaiting tell

    def _new_es(self):
        if self.x0 is not None and self.es is None:
            m = np.array(self.x0)
        elif self.restart_from_best and self.best_z is not None and self.n_restarts % 2 == 1:
            m = np.array(self.best_z)
        else:
            m = np.zeros(self.dim)
        opts = {"bounds": [list(self.bounds[:, 0]), list(self.bounds[:, 1])],
                "seed": int(self.rng.integers(1, 2**31 - 1)), "verbose": -9,
                "tolfun": 1e-9, "tolx": 1e-9}
        if self.popsize:
            opts["popsize"] = int(self.popsize)
        self.es = cma.CMAEvolutionStrategy(list(m), self.sigma0, opts)

    def _new_generation(self):
        if self.es is None:
            self._new_es()
        elif self.es.stop():
            self.n_restarts += 1
            self.popsize = int((self.popsize or self.es.popsize) * self.inc_popsize)
            self._new_es()
        Z = self.es.ask()
        self.gen_Z = [np.asarray(z, float) for z in Z]
        self.gen_X = [np.clip(self.center + self.T @ z, self.xlo, self.xhi) for z in self.gen_Z]
        self.gen_y = {}
        self.queue = list(range(len(self.gen_X)))

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                if self.gen_X and len(self.gen_y) + len(self.handed) < len(self.gen_X):
                    break
                if self.handed:
                    break  # wait for outstanding points of this generation
                self._new_generation()
            i = self.queue.pop(0)
            self.handed.append(i)
            out.append([float(v) for v in self.gen_X[i]])
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = np.asarray(x, float)
            v = float(v)
            match = None
            for i in self.handed:
                if np.allclose(self.gen_X[i], x, atol=1e-12):
                    match = i
                    break
            if v < self.best_y:
                self.best_y, self.best_x = v, x.tolist()
                if match is not None:
                    self.best_z = self.gen_Z[match].tolist()
            if match is not None:
                self.handed.remove(match)
                self.gen_y[match] = v
        if self.gen_X and len(self.gen_y) == len(self.gen_X):
            idx = sorted(self.gen_y)
            self.es.tell([self.gen_Z[i] for i in idx], [self.gen_y[i] for i in idx])
            self.gen_X, self.gen_y = [], {}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"self": self.__dict__, "np": np.random.get_state()}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(d["self"])
        np.random.set_state(d["np"])
        return obj
