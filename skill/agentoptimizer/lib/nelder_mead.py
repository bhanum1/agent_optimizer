"""Nelder-Mead simplex, native ask/tell, so `continue` resumes the exact simplex.

args:
  x0        start vertex in real coordinates. Default: the best point already
            told, else the box center.
  step      initial simplex edge as a fraction of the box width (default 0.1).
  adaptive  Gao-Han dimension-adapted coefficients (default true when dim > 2).
  xtol, ftol  finish when the simplex diameter (unit cube) and the spread of
            vertex values both fall below these (defaults 1e-8, 1e-10).
Points are clipped to the box. Each step costs 1-2 evaluations, a shrink
costs dim.
"""
import numpy as np

from _base import Base


class Optimizer(Base):
    max_ask = 1  # proposes one point at a time; refits or steps on each result
    def __init__(self, space, seed=0, x0=None, step=0.1, adaptive=None, xtol=1e-8, ftol=1e-10):
        super().__init__(space, seed)
        self.x0 = None if x0 is None else self.to_unit(x0)
        n = self.dim
        if adaptive if adaptive is not None else n > 2:
            self.a, self.b, self.c, self.d = 1, 1 + 2 / n, 0.75 - 1 / (2 * n), 1 - 1 / n
        else:
            self.a, self.b, self.c, self.d = 1, 2, 0.5, 0.5
        self.step, self.xtol, self.ftol = step, xtol, ftol
        self.S, self.F = [], []      # simplex vertices and values
        self.queue = []              # vertices still to evaluate (init or shrink)
        self.phase, self.trial = "init", None

    def propose(self):
        if self.phase == "init" and not self.S and not self.queue:
            x0 = self.x0 if self.x0 is not None else self.best()[0] if len(self.y) else np.full(self.dim, 0.5)
            self.queue = [np.clip(x0, 0, 1)]
            for i in range(self.dim):
                e = x0.copy()
                e[i] = e[i] + self.step if e[i] + self.step <= 1 else e[i] - self.step
                self.queue.append(e)
        if self.queue:
            return self.queue[0]
        if self.converged():
            return None
        order = np.argsort(self.F)
        self.S, self.F = [self.S[i] for i in order], [self.F[i] for i in order]
        self.centroid = np.mean(self.S[:-1], axis=0)
        if self.phase in ("init", "shrink", "reflect_done"):
            self.phase = "reflect"
        if self.phase == "reflect":
            self.trial = np.clip(self.centroid + self.a * (self.centroid - self.S[-1]), 0, 1)
        elif self.phase == "expand":
            self.trial = np.clip(self.centroid + self.b * (self.reflected - self.centroid), 0, 1)
        elif self.phase == "outside":
            self.trial = np.clip(self.centroid + self.c * (self.reflected - self.centroid), 0, 1)
        elif self.phase == "inside":
            self.trial = np.clip(self.centroid - self.c * (self.reflected - self.centroid), 0, 1)
        return self.trial

    def observe(self, u, value):
        if self.queue:
            self.queue.pop(0)
            if self.phase == "shrink":
                i = self.shrinking.pop(0)
                self.S[i], self.F[i] = u, value
            else:
                self.S.append(u)
                self.F.append(value)
            return
        f = value
        if self.phase == "reflect":
            self.reflected, self.fr = u, f
            if self.F[0] <= f < self.F[-2]:
                self.accept(u, f)
            elif f < self.F[0]:
                self.phase = "expand"
            elif f < self.F[-1]:
                self.phase = "outside"
            else:
                self.phase = "inside"
        elif self.phase == "expand":
            self.accept(*((u, f) if f < self.fr else (self.reflected, self.fr)))
        elif self.phase == "outside":
            self.accept(u, f) if f <= self.fr else self.shrink()
        elif self.phase == "inside":
            self.accept(u, f) if f < self.F[-1] else self.shrink()

    def accept(self, u, f):
        self.S[-1], self.F[-1] = u, f
        self.phase = "reflect_done"

    def shrink(self):
        self.phase = "shrink"
        best = self.S[0]
        self.queue = [best + self.d * (self.S[i] - best) for i in range(1, len(self.S))]
        self.shrinking = list(range(1, len(self.S)))

    def converged(self):
        S = np.array(self.S)
        return (np.max(np.abs(S - S[0])) <= self.xtol and max(self.F) - min(self.F) <= self.ftol)
