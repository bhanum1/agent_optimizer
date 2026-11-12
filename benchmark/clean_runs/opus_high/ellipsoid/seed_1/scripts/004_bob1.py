import pickle
import warnings
import logging
import numpy as np
import pybobyqa


class _Need(Exception):
    def __init__(self, x):
        self.x = x


def _key(x):
    return tuple(np.round(np.asarray(x, dtype=float), 12))


class Optimizer:
    """Py-BOBYQA served one point at a time by deterministic replay.

    Each ask re-runs the current solve from its start with a cache of measured values;
    the first uncached point it requests is the next point. When a solve finishes, a
    new one starts from the best point so far with rhobeg shrunk by `shrink`.
    """

    max_ask = 1

    def __init__(self, space, rhobeg=0.1, rhoend=1e-8, shrink=0.5, rhobeg_min=1e-3):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.rhobeg = float(rhobeg)
        self.rhoend = float(rhoend)
        self.shrink = float(shrink)
        self.rhobeg_min = float(rhobeg_min)
        self.cache = {}
        self.best_x = None
        self.best_y = np.inf
        self.run_x0 = None       # start of the current solve
        self.run_rho = self.rhobeg
        self.runs = 0

    def _solve(self):
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]

        def f(x):
            x = np.clip(np.asarray(x, dtype=float), lb, ub)
            k = _key(x)
            if k in self.cache:
                return self.cache[k]
            raise _Need(x)

        rho = min(self.run_rho, 0.499 * float(np.min(ub - lb)))
        logging.disable(logging.CRITICAL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pybobyqa.solve(f, self.run_x0.copy(), bounds=(lb, ub), rhobeg=rho,
                               rhoend=self.rhoend, maxfun=10**6,
                               seek_global_minimum=False, objfun_has_noise=False)
            except _Need as e:
                return e.x
            finally:
                logging.disable(logging.NOTSET)
        return None

    def ask(self, n):
        for _ in range(50):
            if self.run_x0 is None:
                self.run_x0 = (self.best_x.copy() if self.best_x is not None
                               else self.bounds.mean(axis=1))
            x = self._solve()
            if x is not None:
                return [x.tolist()]
            # solve finished: restart from the best point with a smaller rhobeg
            self.runs += 1
            self.run_rho = max(self.run_rho * self.shrink, self.rhobeg_min)
            self.run_x0 = self.best_x.copy()
        return []

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = np.clip(np.asarray(x, dtype=float), self.bounds[:, 0], self.bounds[:, 1])
            v = float(v)
            self.cache[_key(x)] = v
            if v < self.best_y:
                self.best_y, self.best_x = v, x.copy()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
