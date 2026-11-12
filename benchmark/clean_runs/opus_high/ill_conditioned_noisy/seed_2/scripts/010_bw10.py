import json
import numpy as np
import pybobyqa


class _Need(Exception):
    def __init__(self, z):
        self.z = z


class Optimizer:
    """Py-BOBYQA in whitened coordinates x = clip(center + T z), driven by deterministic
    replay. Each ask re-runs the solver from scratch with np.random seeded, answering its
    calls from the stored history, and stops at the first call with no stored value.
    When a run terminates, a new run starts from the best z seen in that run, with rhobeg
    multiplied by `shrink` but not below `rhobeg_min`. Warm-start points are ignored."""
    max_ask = 1

    def __init__(self, space, center, T, rhobeg=4.0, rhoend=1.0, seed=0,
                 maxfun_run=150, shrink=0.7, rhobeg_min=2.0, zbox=60.0, npt=None,
                 noisy=False, max_runs=50):
        self.lo = [b[0] for b in space["bounds"]]
        self.hi = [b[1] for b in space["bounds"]]
        self.dim = space["dim"]
        self.center = [float(v) for v in center]
        self.T = [[float(v) for v in row] for row in T]
        self.rhobeg, self.rhoend = float(rhobeg), float(rhoend)
        self.seed, self.maxfun_run, self.shrink = int(seed), int(maxfun_run), float(shrink)
        self.rhobeg_min, self.zbox, self.max_runs = float(rhobeg_min), float(zbox), int(max_runs)
        self.npt = int(npt) if npt else (self.dim + 1) * (self.dim + 2) // 2
        self.noisy = bool(noisy)
        self.hist_z, self.hist_y = [], []
        self.pending = None
        self.pending_z = None

    def _x(self, z):
        x = np.array(self.center) + np.array(self.T) @ np.array(z, float)
        return np.clip(x, self.lo, self.hi)

    def _replay(self):
        zlo = -self.zbox * np.ones(self.dim)
        zhi = self.zbox * np.ones(self.dim)
        k = 0
        run = 0
        z0 = np.zeros(self.dim)
        rhobeg = self.rhobeg
        while True:
            np.random.seed(self.seed + 1000 * run)
            start = k

            def f(z):
                nonlocal k
                if k < len(self.hist_y):
                    v = self.hist_y[k]
                    k += 1
                    return v
                raise _Need(np.array(z, float))

            try:
                pybobyqa.solve(f, z0, bounds=(zlo, zhi), npt=self.npt,
                               rhobeg=rhobeg, rhoend=self.rhoend, maxfun=self.maxfun_run,
                               objfun_has_noise=self.noisy, scaling_within_bounds=False)
            except _Need as e:
                return e.z
            if k == start:
                return None
            seg = np.array(self.hist_y[start:k])
            z0 = np.array(self.hist_z[start + int(np.argmin(seg))], float)
            rhobeg = max(rhobeg * self.shrink, self.rhobeg_min)
            run += 1
            if run > self.max_runs:
                return None

    def ask(self, n):
        if self.pending is None:
            z = self._replay()
            if z is None:
                return []
            self.pending_z = [float(v) for v in z]
            self.pending = [float(v) for v in self._x(z)]
        return [list(self.pending)]

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            if self.pending is not None and np.allclose(x, self.pending, atol=1e-12):
                self.hist_z.append(list(self.pending_z))
                self.hist_y.append(float(v))
                self.pending = None
                self.pending_z = None

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path) as fh:
            obj.__dict__.update(json.load(fh))
        return obj
