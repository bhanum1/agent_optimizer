import json
import numpy as np
import pybobyqa


class _NeedPoint(Exception):
    def __init__(self, x):
        self.x = x


def _key(x):
    return tuple(np.round(np.asarray(x, float), 10))


class Optimizer:
    """BOBYQA (pybobyqa) as ask/tell, by deterministic replay over a cache of
    measured points. A chain of solves: each starts from the best point seen so
    far (in replay order); rhobeg cycles through `rhobegs`."""

    def __init__(self, space, rhobegs=(0.1, 0.03, 0.3), rhoend=1e-9, npt=21,
                 max_solves=200):
        self.lo = [b[0] for b in space["bounds"]]
        self.hi = [b[1] for b in space["bounds"]]
        self.rhobegs = list(rhobegs)
        self.rhoend = rhoend
        self.npt = npt
        self.max_solves = max_solves
        self.warm = []      # [x, y] told before the first ask
        self.cache = {}     # key -> y
        self.pending = []   # points asked, not told
        self.asked_any = False

    def _replay(self):
        lo, hi = np.array(self.lo), np.array(self.hi)
        seen = [(np.array(x), y) for x, y in self.warm]
        def f(x):
            x = np.clip(np.asarray(x, float), lo, hi)
            k = _key(x)
            if k not in self.cache:
                raise _NeedPoint(x)
            seen.append((x, self.cache[k]))
            return self.cache[k]
        for s in range(self.max_solves):
            if seen:
                x0 = min(seen, key=lambda t: t[1])[0]
            else:
                x0 = (lo + hi) / 2
            rb = self.rhobegs[s % len(self.rhobegs)]
            np.random.seed(s)
            try:
                pybobyqa.solve(f, x0.copy(), bounds=(lo, hi), npt=self.npt,
                               rhobeg=rb, rhoend=self.rhoend, maxfun=100000,
                               do_logging=False)
            except _NeedPoint as e:
                return e.x
        return None

    def ask(self, n):
        self.asked_any = True
        if self.pending:
            return []  # one point at a time; wait for its value
        x = self._replay()
        if x is None:
            return []
        x = [float(v) for v in x]
        self.pending.append(x)
        return [x]

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            self.cache[_key(x)] = float(v)
            if not self.asked_any:
                self.warm.append([[float(t) for t in x], float(v)])
        keys = {_key(x) for x in X}
        self.pending = [p for p in self.pending if _key(p) not in keys]

    def save(self, path):
        d = dict(self.__dict__)
        d["cache"] = [[list(k), v] for k, v in self.cache.items()]
        with open(path, "w") as fh:
            json.dump(d, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        o = cls.__new__(cls)
        d["cache"] = {tuple(k): v for k, v in d["cache"]}
        o.__dict__.update(d)
        return o
