"""Scrambled Sobol sequence (scipy.stats.qmc.Sobol). Space-filling initial
designs; points are balanced in powers of two. Box problems."""
from scipy.stats import qmc

from _base import Base


class Optimizer(Base):
    def __init__(self, space, seed=0, scramble=True):
        super().__init__(space, seed)
        self.sampler = qmc.Sobol(self.dim, scramble=scramble, seed=seed)

    def propose(self):
        return self.sampler.random(1)[0]
