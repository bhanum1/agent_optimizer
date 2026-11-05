"""Differential evolution (Storn & Price), steady-state, in the unit cube.

args:
  pop_size  population size (default 10 * dim, at least 5).
  F         differential weight (default 0.8).
  CR        crossover rate (default 0.9).
  strategy  "rand1bin" (default) or "best1bin".
The first pop_size evaluations fill the population; points already told
(e.g. warm_start) fill it first, best first. Then each evaluation is one
trial vector against one target, cycling through the population; the trial
replaces the target if it is no worse. Out-of-box components are resampled
between the base vector and the bound.
"""
import numpy as np

from _base import Base


class Optimizer(Base):
    def __init__(self, space, seed=0, pop_size=None, F=0.8, CR=0.9, strategy="rand1bin"):
        super().__init__(space, seed)
        if strategy not in ("rand1bin", "best1bin"):
            raise ValueError("strategy must be rand1bin or best1bin")
        self.pop_size = max(5, pop_size or 10 * self.dim)
        self.F, self.CR, self.strategy = F, CR, strategy
        self.pop, self.fit, self.target = [], [], 0

    def absorb(self, u, value):
        if len(self.pop) < self.pop_size:
            self.observe(u, value)

    def propose(self):
        if len(self.pop) < self.pop_size:
            return self.rng.random(self.dim)
        P, i = np.array(self.pop), self.target
        others = [j for j in range(self.pop_size) if j != i]
        r1, r2, r3 = self.rng.choice(others, 3, replace=False)
        base = P[int(np.argmin(self.fit))] if self.strategy == "best1bin" else P[r1]
        mutant = base + self.F * (P[r2] - P[r3])
        low, high = mutant < 0, mutant > 1
        mutant[low] = base[low] * self.rng.random(low.sum())
        mutant[high] = base[high] + (1 - base[high]) * self.rng.random(high.sum())
        cross = self.rng.random(self.dim) < self.CR
        cross[self.rng.integers(self.dim)] = True
        return np.where(cross, mutant, P[i])

    def observe(self, u, value):
        if len(self.pop) < self.pop_size:
            self.pop.append(u)
            self.fit.append(value)
            return
        if value <= self.fit[self.target]:
            self.pop[self.target], self.fit[self.target] = u, value
        self.target = (self.target + 1) % self.pop_size

    def tell(self, X, y, g):
        if len(X) > 1:  # warm start: fill the population best first
            order = np.argsort([float(v) for v in y])
            X, y, g = [X[i] for i in order], [y[i] for i in order], [g[i] for i in order]
        super().tell(X, y, g)
