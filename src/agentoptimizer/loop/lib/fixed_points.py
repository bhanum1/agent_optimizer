"""Evaluate a list of points you choose, in order, then finish.

args: points  list of points in real coordinates, or candidate indices
              for an indexed pool. A block longer than the list ends early.
Use it for probes, replicates, grids, or a design computed in work/.
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
        return int(p) if self.indexed else self.to_unit(np.asarray(p, dtype=float))
