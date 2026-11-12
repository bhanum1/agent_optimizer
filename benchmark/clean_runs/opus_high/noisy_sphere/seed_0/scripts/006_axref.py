import json
import numpy as np


class Optimizer:
    """Iterated axial refinement: each round measures the centre and centre±h on every
    axis, moves the centre to the per-axis quadratic minimiser, then shrinks h."""
    max_ask = 1

    def __init__(self, space, center, h=0.03, shrink=3.0, hmin=1e-4):
        self.lo = [b[0] for b in space["bounds"]]
        self.hi = [b[1] for b in space["bounds"]]
        self.c = [float(v) for v in center]
        self.h, self.shrink, self.hmin = float(h), float(shrink), float(hmin)
        self.plan, self.res, self.log = [], [], []
        self._new_round()

    def _new_round(self):
        d = len(self.c)
        self.plan = [list(self.c)]
        for i in range(d):
            for s in (self.h, -self.h):
                p = list(self.c); p[i] = min(max(p[i] + s, self.lo[i]), self.hi[i])
                self.plan.append(p)
        self.res = [None] * len(self.plan)
        self.next = 0

    def _finish_round(self):
        c = np.array(self.c); f0 = self.res[0]; h = self.h
        shift = np.zeros(len(c))
        for i in range(len(c)):
            yp, ym = self.res[1 + 2 * i], self.res[2 + 2 * i]
            a = (yp + ym - 2 * f0) / (2 * h * h)
            if a > 0:
                shift[i] = np.clip(-(yp - ym) / (2 * h) / (2 * a), -2 * h, 2 * h)
        self.log.append({"c": self.c, "h": h, "f0": f0, "shift": shift.tolist()})
        self.c = np.clip(c + shift, self.lo, self.hi).tolist()
        self.h = max(h / self.shrink, self.hmin)
        self._new_round()

    def ask(self, n):
        out = []
        while len(out) < n and self.next < len(self.plan):
            out.append(list(self.plan[self.next])); self.next += 1
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            for k in range(len(self.plan)):
                if self.res[k] is None and np.allclose(self.plan[k], x, atol=1e-12):
                    self.res[k] = float(v); break
        if all(r is not None for r in self.res):
            self._finish_round()

    def save(self, path):
        json.dump(self.__dict__, open(path, "w"))

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls); o.__dict__.update(json.load(open(path))); return o
