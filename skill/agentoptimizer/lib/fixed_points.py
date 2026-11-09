"""Evaluate a list of points you choose, in order, then finish.

args: points  list of points: encoded points (lists of numbers), configurations
              ({variable name: value}, for any problem), or candidate indices for
              an indexed pool. A block longer than the list ends early.
Use it for probes, grids, a design computed in work/, or replicates: on a noisy
problem, listing an already-measured configuration requests a replicate of it.
"""
import numpy as np

from _base import Base


class Optimizer(Base):
    def __init__(self, space, seed=0, points=()):
        super().__init__(space, seed)
        self.queue = list(points)

    def propose(self):
        if not self.queue:
            return None
        p = self.queue.pop(0)
        if self.indexed and not isinstance(p, dict):
            return int(p)
        x = self.encode_config(p) if isinstance(p, dict) else np.asarray(p, dtype=float)
        return self.to_unit(x)
