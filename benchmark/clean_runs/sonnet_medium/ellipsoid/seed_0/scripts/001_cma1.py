import pickle
import numpy as np
import cma
from scipy.stats import qmc


class Optimizer:
    def __init__(self, space, seed=0, n_sobol=64, sigma0=0.25, popsize=None, warm_sigma=None):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lo)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sigma0 = sigma0
        self.pop0 = popsize
        self.restarts = 0
        pts = qmc.Sobol(self.d, seed=seed).random(n_sobol)
        self.init_queue = [list(self.lo + p * (self.hi - self.lo)) for p in pts]
        self.hist = []  # (u, y) in unit cube
        self.es = None
        self.gen = []  # outstanding unit-cube solutions
        self.gen_res = {}
        self.out_init = []

    def _u(self, x):
        return (np.asarray(x, float) - self.lo) / (self.hi - self.lo)

    def _new_es(self):
        pop = self.pop0 or (4 + int(3 * np.log(self.d)))
        pop = int(pop * (2 ** (self.restarts // 2)))
        if self.restarts == 0 and self.hist:
            u0 = min(self.hist, key=lambda t: t[1])[0]
        else:
            u0 = self.rng.random(self.d)
        s = self.sigma0 if self.restarts % 2 == 0 else self.sigma0 * 0.4
        self.es = cma.CMAEvolutionStrategy(
            list(u0), s,
            {"bounds": [0, 1], "popsize": pop, "seed": int(self.rng.integers(1, 2**31 - 1)),
             "verbose": -9, "tolfun": 1e-12, "tolx": 1e-10})
        self.restarts += 1

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.init_queue:
                x = self.init_queue.pop(0)
                self.out_init.append(x)
                out.append([float(v) for v in x])
                continue
            if self.es is None or self.es.stop():
                self._new_es()
            if not self.gen:
                self.gen = [np.array(s) for s in self.es.ask()]
                self.gen_res = {}
            # next unissued member
            k = next((i for i in range(len(self.gen)) if i not in self.gen_res and i not in getattr(self, "issued", set())), None)
            if k is None:
                # all issued but not told; reissue the first untold
                k = next(i for i in range(len(self.gen)) if i not in self.gen_res)
            else:
                self.issued = getattr(self, "issued", set()) | {k}
            u = np.clip(self.gen[k], 0, 1)
            out.append([float(v) for v in self.lo + u * (self.hi - self.lo)])
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            u = self._u(x)
            self.hist.append((u, float(v)))
            if self.es is not None and self.gen:
                for i, s in enumerate(self.gen):
                    if i not in self.gen_res and np.allclose(np.clip(s, 0, 1), u, atol=1e-9):
                        self.gen_res[i] = float(v)
                        break
                if len(self.gen_res) == len(self.gen):
                    self.es.tell([list(s) for s in self.gen], [self.gen_res[i] for i in range(len(self.gen))])
                    self.gen = []
                    self.gen_res = {}
                    self.issued = set()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
