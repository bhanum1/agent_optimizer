"""Particle swarm optimization (inertia-weight PSO), in the unit cube.

args:
  swarm  number of particles (default 10 + 2 sqrt(dim), rounded).
  w      inertia weight (default 0.7298).
  c1     cognitive coefficient (default 1.49618).
  c2     social coefficient (default 1.49618).
  vmax   velocity clamp as a fraction of the box width (default 0.2).
The first `swarm` evaluations place the particles (points already told fill
first). Then particles move one at a time, cycling through the swarm;
positions are clipped to the box and velocity is zeroed on the clipped axes.
"""
import numpy as np

from _base import Base


class Optimizer(Base):
    def __init__(self, space, seed=0, swarm=None, w=0.7298, c1=1.49618, c2=1.49618, vmax=0.2):
        super().__init__(space, seed)
        self.n = swarm or int(round(10 + 2 * np.sqrt(self.dim)))
        self.w, self.c1, self.c2, self.vmax = w, c1, c2, vmax
        self.pos, self.vel, self.pbest, self.pfit = [], [], [], []
        self.k = 0

    def absorb(self, u, value):
        if len(self.pos) < self.n:
            self.observe(u, value)

    def propose(self):
        if len(self.pos) < self.n:
            return self.rng.random(self.dim)
        k, g = self.k, self.pbest[int(np.argmin(self.pfit))]
        r1, r2 = self.rng.random(self.dim), self.rng.random(self.dim)
        v = self.w * self.vel[k] + self.c1 * r1 * (self.pbest[k] - self.pos[k]) + self.c2 * r2 * (g - self.pos[k])
        v = np.clip(v, -self.vmax, self.vmax)
        x = self.pos[k] + v
        v[(x < 0) | (x > 1)] = 0
        self.vel[k] = v
        return np.clip(x, 0, 1)

    def observe(self, u, value):
        if len(self.pos) < self.n:
            self.pos.append(u)
            self.vel.append(self.rng.uniform(-self.vmax, self.vmax, self.dim))
            self.pbest.append(u)
            self.pfit.append(value)
            return
        k = self.k
        self.pos[k] = u
        if value < self.pfit[k]:
            self.pbest[k], self.pfit[k] = u, value
        self.k = (k + 1) % self.n
