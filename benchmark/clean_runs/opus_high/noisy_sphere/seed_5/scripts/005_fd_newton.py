"""Finite-difference diagonal Newton refinement for a smooth, low-noise bowl.

Each iteration measures the 2n central-difference stencil c +- h e_i around the
current centre c. From it, it estimates the gradient g_i = (f+ - f-)/(2h) and the
diagonal curvature H_ii = (f+ + f- - 2 f0)/h^2. It then measures the diagonal Newton
point c - g/H, with each coordinate clipped to `max_step`. If that point beats f0, it
becomes the new centre. Otherwise the step is halved and retried, up to `max_halve`
times. After `iters` iterations, every ask returns the centre (replicates).
"""
import pickle

import numpy as np


class Optimizer:
    max_ask = 1

    def __init__(self, space, center, f0=None, h=0.01, iters=3, max_step=0.05,
                 max_halve=2, h_fallback=2.0):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.dim = len(b)
        self.c = np.array(center, float)
        self.f0 = f0
        self.h, self.iters, self.max_step = h, iters, max_step
        self.max_halve, self.h_fallback = max_halve, h_fallback
        self.queue = []      # (kind, point) still to ask
        self.asked = []      # (kind, point) asked, not yet told
        self.stencil = {}    # index -> value
        self.step = None
        self.halves = 0
        self.it = 0
        self.history = []

    def _plan(self):
        if self.f0 is None:
            self.queue.append(("center", self.c.copy()))
        elif self.it >= self.iters:
            self.queue.append(("rep", self.c.copy()))
        elif self.step is None and len(self.stencil) < 2 * self.dim:
            for i in range(self.dim):
                for s in (1, -1):
                    if 2 * i + (s < 0) not in self.stencil:
                        p = self.c.copy()
                        p[i] += s * self.h
                        self.queue.append((("st", 2 * i + (s < 0)), p))
        else:
            if self.step is None:
                fp = np.array([self.stencil[2 * i] for i in range(self.dim)])
                fm = np.array([self.stencil[2 * i + 1] for i in range(self.dim)])
                g = (fp - fm) / (2 * self.h)
                H = (fp + fm - 2 * self.f0) / self.h ** 2
                H = np.where(H > 0.1, H, self.h_fallback)
                self.step = np.clip(-g / H, -self.max_step, self.max_step)
                self.history.append({"it": self.it, "g": g.tolist(), "H": H.tolist()})
            self.queue.append(("step", self.c + self.step))

    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.queue:
                self._plan()
            kind, p = self.queue.pop(0)
            p = np.clip(p, self.lo, self.hi)
            self.asked.append((kind, p))
            out.append([float(v) for v in p])
        return out

    def _next_iter(self):
        self.it += 1
        self.stencil, self.step, self.halves = {}, None, 0
        self.queue = []

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x, v = np.array(x, float), float(v)
            match = [k for k, (_, p) in enumerate(self.asked) if np.allclose(p, x, atol=1e-12)]
            if not match:
                continue  # warm-start data is not used
            kind, p = self.asked.pop(match[0])
            if kind == "center":
                self.f0 = v
            elif kind == "rep":
                pass
            elif kind == "step":
                self.history.append({"it": self.it, "step": self.step.tolist(), "f": v, "f0": self.f0})
                if v < self.f0:
                    self.c, self.f0 = p, v
                    self._next_iter()
                elif self.halves < self.max_halve:
                    self.halves += 1
                    self.step = self.step / 2
                else:
                    self._next_iter()
            else:
                self.stencil[kind[1]] = v

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
