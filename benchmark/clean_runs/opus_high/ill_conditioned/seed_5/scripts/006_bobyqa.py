import pickle
import numpy as np
import pybobyqa


class _NeedPoint(Exception):
    def __init__(self, x):
        self.x = x


class Optimizer:
    """Py-BOBYQA run as ask/tell by replay: the objective is deterministic, so each ask
    reruns BOBYQA (fixed seed) on the cached values until it requests an unknown point,
    which is the point asked. When a run terminates, a new run starts from the best
    point seen, rhobeg doubled per restart (capped at 1)."""

    def __init__(self, space, seed=0, rhobeg=0.05, rhoend=1e-7, x0=None, maxfun=5000):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.seed, self.rhobeg, self.rhoend, self.maxfun = seed, rhobeg, rhoend, maxfun
        self.x0 = None if x0 is None else np.asarray(x0, float)
        self.cache = {}      # key -> value
        self.runs = []       # (x0, seed, best_x) of finished runs
        self.best_x, self.best_y = None, np.inf

    def _key(self, x): return tuple(np.round(np.asarray(x, float), 12))

    def _replay(self, x0, seed, rhobeg):
        def f(x):
            x = np.clip(x, self.lo, self.hi)
            k = self._key(x)
            if k in self.cache: return self.cache[k]
            raise _NeedPoint(x)
        np.random.seed(seed)
        try:
            pybobyqa.solve(f, x0.copy(), bounds=(self.lo, self.hi), rhobeg=rhobeg,
                           rhoend=self.rhoend, maxfun=self.maxfun, objfun_has_noise=False)
        except _NeedPoint as e:
            return e.x
        return None

    def _current_start(self):
        if not self.runs:
            x0 = self.x0 if self.x0 is not None else self.best_x
            if x0 is None: x0 = (self.lo + self.hi) / 2
            return np.clip(x0, self.lo + 1e-9, self.hi - 1e-9), self.seed, self.rhobeg
        return None

    def ask(self, n):
        if not hasattr(self, "_cur"): self._cur = None
        for _ in range(10):  # at most a few consecutive finished runs
            if self._cur is None:
                start = self._current_start()
                self._cur = start if start is not None else (np.array(self.runs[-1][2]), self.seed + len(self.runs),
                                                                 min(1.0, self.rhobeg * 2 ** len(self.runs)))
            x = self._replay(*self._cur)
            if x is None:  # run finished: restart from best seen
                self.runs.append((self._cur[0].tolist(), self._cur[1], self.best_x.tolist()))
                self._cur = None
                continue
            return [x.tolist()]  # sequential: one outstanding point at a time
        return []

    max_ask = 1

    def tell(self, X, y, g=None):
        for x, v in zip(X, y):
            v = float(v)
            if not np.isfinite(v): v = 1e300
            self.cache[self._key(x)] = v
            if v < self.best_y: self.best_y, self.best_x = v, np.asarray(x, float)

    def save(self, path):
        with open(path, "wb") as fh: pickle.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as fh: o.__dict__.update(pickle.load(fh))
        return o
