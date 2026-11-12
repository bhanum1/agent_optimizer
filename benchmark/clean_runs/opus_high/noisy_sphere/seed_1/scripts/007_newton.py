"""Iterated Newton refinement with central differences on a near-quadratic basin.

Each iteration measures a center c and c +- h e_i, estimates the gradient by central
differences, and moves to c - g/a, using the fixed second derivatives a (args) unless
the difference estimate is reliable (h large enough). h shrinks with the last step.
"""
import json
import numpy as np


class Optimizer:
    def __init__(self, space, center=None, a=None, h0=0.002, hfactor=1.0, min_h=2e-7,
                 update_a=False):
        self.lo = [b[0] for b in space["bounds"]]
        self.hi = [b[1] for b in space["bounds"]]
        self.d = len(self.lo)
        self.c = list(center)
        self.a = list(a) if a is not None else [2.0] * self.d
        self.h = float(h0)
        self.hfactor = float(hfactor)
        self.min_h = float(min_h)
        self.update_a = bool(update_a)
        self.plan = []        # points of the current iteration, in order
        self.vals = []        # values told for the plan, by index
        self.next_i = 0       # next plan index to ask
        self.best = None      # (y, x)
        self.history = []     # (center, h, step norm, f0)
        self._new_iter()

    def _clip(self, x):
        return [min(max(v, l), u) for v, l, u in zip(x, self.lo, self.hi)]

    def _new_iter(self):
        c = self.c
        plan = [list(c)]
        for i in range(self.d):
            for s in (1, -1):
                p = list(c)
                p[i] = c[i] + s * self.h
                plan.append(self._clip(p))
        self.plan = plan
        self.vals = [None] * len(plan)
        self.next_i = 0

    def _finish_iter(self):
        f = self.vals
        f0 = f[0]
        g = []
        for i in range(self.d):
            xp, xm = self.plan[1 + 2 * i][i], self.plan[2 + 2 * i][i]
            fp, fm = f[1 + 2 * i], f[2 + 2 * i]
            g.append((fp - fm) / (xp - xm))
            if self.update_a:
                hh = (xp - xm) / 2
                ai = (fp + fm - 2 * f0) / hh ** 2
                if 1.0 < ai < 4.0:
                    self.a[i] = ai
        step = [-gi / ai for gi, ai in zip(g, self.a)]
        sn = float(np.linalg.norm(step))
        self.history.append([list(self.c), self.h, sn, f0])
        newc = self._clip([ci + si for ci, si in zip(self.c, step)])
        # guard: never move further than 3x the previous probe scale
        if sn > 3 * max(self.h, 1e-12) * np.sqrt(self.d):
            newc = list(self.best[1])
        self.c = newc
        self.h = max(self.hfactor * sn, self.min_h)
        self._new_iter()

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.next_i >= len(self.plan):
                break
            out.append(list(self.plan[self.next_i]))
            self.next_i += 1
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v)
            if self.best is None or v < self.best[0]:
                self.best = (v, list(x))
            open_j = [j for j in range(self.next_i) if self.vals[j] is None]
            hit = [j for j in open_j
                   if all(abs(a - b) < 1e-9 for a, b in zip(self.plan[j], x))]
            if hit:
                self.vals[hit[0]] = v
        if all(v is not None for v in self.vals):
            self._finish_iter()

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path) as fh:
            st = json.load(fh)
        st["best"] = tuple(st["best"]) if st["best"] is not None else None
        obj.__dict__.update(st)
        return obj
