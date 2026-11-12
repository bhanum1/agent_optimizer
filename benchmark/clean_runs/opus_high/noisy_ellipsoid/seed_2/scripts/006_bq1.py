import json
import numpy as np
import pybobyqa


class _Need(Exception):
    def __init__(self, x):
        self.x = x


class Optimizer:
    """Py-BOBYQA (full quadratic npt, noise mode optional) driven by deterministic replay.
    Each ask re-runs the solver from scratch with np.random seeded, answering its
    calls from the stored history, and stops at the first call with no stored value.
    When a run terminates, a new run starts from the best point seen, with rhobeg
    shrunk by `shrink`. Warm-start points only set x0 (best) if x0 is not given."""
    max_ask = 1

    def __init__(self, space, x0=None, rhobeg=0.02, rhoend=1e-5, seed=0,
                 maxfun_run=400, shrink=0.5, npt=None, noisy=False):
        self.lo = [b[0] for b in space["bounds"]]
        self.hi = [b[1] for b in space["bounds"]]
        self.dim = space["dim"]
        self.x0 = None if x0 is None else [float(v) for v in x0]
        self.rhobeg, self.rhoend = float(rhobeg), float(rhoend)
        self.seed, self.maxfun_run, self.shrink = int(seed), int(maxfun_run), float(shrink)
        self.npt = int(npt) if npt else (self.dim + 1) * (self.dim + 2) // 2
        self.noisy = bool(noisy)
        self.hist_x, self.hist_y = [], []
        self.pending = None
        self.ext_best = (None, np.inf)
        self.done = False

    def _replay(self):
        lo, hi = np.array(self.lo), np.array(self.hi)
        k = 0
        run = 0
        x0 = np.array(self.x0 if self.x0 is not None else self.ext_best[0], float)
        rhobeg = self.rhobeg
        while True:
            np.random.seed(self.seed + 1000 * run)
            start = k

            def f(x):
                nonlocal k
                if k < len(self.hist_y):
                    v = self.hist_y[k]
                    k += 1
                    return v
                raise _Need(np.clip(np.array(x, float), lo, hi))

            try:
                pybobyqa.solve(f, np.clip(x0, lo, hi), bounds=(lo, hi), npt=self.npt,
                               rhobeg=rhobeg, rhoend=self.rhoend, maxfun=self.maxfun_run,
                               objfun_has_noise=self.noisy, scaling_within_bounds=False)
            except _Need as e:
                return e.x
            if k == start:
                return None
            seg = np.array(self.hist_y[start:k])
            x0 = np.array(self.hist_x[start + int(np.argmin(seg))], float)
            rhobeg = max(rhobeg * self.shrink, 20 * self.rhoend)
            run += 1
            if run > 50:
                return None

    def ask(self, n):
        if self.pending is None:
            x = self._replay()
            if x is None:
                return []
            self.pending = [float(v) for v in x]
        return [list(self.pending)]

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            if self.pending is not None and np.allclose(x, self.pending, atol=1e-12):
                self.hist_x.append(list(self.pending))
                self.hist_y.append(float(v))
                self.pending = None
            elif float(v) < self.ext_best[1]:
                self.ext_best = ([float(t) for t in x], float(v))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path) as fh:
            obj.__dict__.update(json.load(fh))
        obj.ext_best = tuple(obj.ext_best)
        return obj
