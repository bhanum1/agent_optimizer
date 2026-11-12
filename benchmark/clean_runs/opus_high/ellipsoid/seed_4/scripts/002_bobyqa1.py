"""Py-BOBYQA as ask/tell by deterministic replay.

Each ask re-runs Py-BOBYQA from scratch against a lookup of the points already
measured by this optimizer; the first point not in the lookup is the next ask.
Valid because the objective is deterministic. When a run finishes (rhoend reached
or maxfun), a new run starts from the best point seen with a fresh rhobeg
(shrunk by restart_shrink each time), so later runs refine the incumbent.
"""
import pickle

import numpy as np
import pybobyqa


class _Need(Exception):
    def __init__(self, x):
        self.x = x


class Optimizer:
    max_ask = 1

    def __init__(self, space, x0=None, rhobeg=1.0, rhoend=1e-8, npt=None,
                 maxfun_run=400, restart_shrink=0.5, rhobeg_min=1e-3, seed=0):
        self.lb = np.array([b[0] for b in space["bounds"]], float)
        self.ub = np.array([b[1] for b in space["bounds"]], float)
        self.dim = len(self.lb)
        self.x0 = x0
        self.rhobeg = rhobeg
        self.rhoend = rhoend
        self.npt = npt or 2 * self.dim + 1
        self.maxfun_run = maxfun_run
        self.restart_shrink = restart_shrink
        self.rhobeg_min = rhobeg_min
        self.seed = seed
        self.X, self.y = [], []          # everything told
        self.run = None                  # dict(x0, rhobeg, keys: list of (key,y))
        self.runs = []
        self.asked = None

    def _key(self, x):
        return tuple(np.round(np.asarray(x, float), 12))

    def _new_run(self):
        if self.run is None and self.x0 is not None:
            x0 = np.asarray(self.x0, float)
            rb = self.rhobeg
        else:
            x0 = np.asarray(self.X[int(np.argmin(self.y))]) if self.y else (self.lb + self.ub) / 2
            rb = self.rhobeg if self.run is None else max(self.run["rhobeg"] * self.restart_shrink, self.rhobeg_min)
        rb = min(rb, 0.49 * float(np.min(self.ub - self.lb)))
        self.run = {"x0": x0.tolist(), "rhobeg": rb, "lookup": {}, "start": len(self.y)}
        self.runs.append([len(self.y), rb])

    def _replay(self):
        lookup = self.run["lookup"]

        def f(x):
            k = self._key(x)
            if k in lookup:
                return lookup[k]
            raise _Need(np.asarray(x, float).copy())

        np.random.seed(self.seed)
        try:
            pybobyqa.solve(f, np.array(self.run["x0"]), bounds=(self.lb, self.ub),
                           npt=self.npt, rhobeg=self.run["rhobeg"], rhoend=self.rhoend,
                           maxfun=self.maxfun_run, seek_global_minimum=False,
                           scaling_within_bounds=False, do_logging=False,
                           user_params={"logging.save_diagnostic_info": False})
        except _Need as e:
            return e.x
        return None

    def ask(self, n):
        if self.run is None:
            self._new_run()
        for _ in range(5):
            x = self._replay()
            if x is not None:
                break
            self._new_run()
        x = np.clip(x, self.lb, self.ub)
        self.asked = (self._key(x), x.tolist())
        return [x.tolist()]

    def tell(self, X, y, g=None):
        for xi, yi in zip(X, y):
            self.X.append(list(map(float, xi)))
            self.y.append(float(yi))
            if self.asked is not None and self._key(xi) == self.asked[0] and self.run is not None:
                self.run["lookup"][self.asked[0]] = float(yi)
                self.asked = None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
