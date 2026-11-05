"""Uniform random search: uniform points in the box, or uniformly chosen
unused candidates in an indexed pool. No settings besides the seed."""
from _base import Base


class Optimizer(Base):
    def propose(self):
        if self.indexed:
            free = [i for i in range(len(self.cands)) if i not in self.used and i not in self.pending]
            return int(self.rng.choice(free)) if free else None
        return self.rng.random(self.dim)
