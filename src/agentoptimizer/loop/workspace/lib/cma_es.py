"""CMA-ES (pycma), in the unit cube with box bounds handled by pycma.

args:
  x0       start mean in real coordinates. Default: the best point already
           told (e.g. with warm_start), else the box center.
  sigma0   initial step size as a fraction of the box width (default 0.25).
  popsize  population per generation (default pycma's 4 + 3 ln(dim)).
  options  dict of further pycma options (e.g. {"CMA_diagonal": true}).
Updates after each full generation. Finishes when pycma's stop criteria fire.
Warm-start points only choose x0; they do not update the distribution.
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Could not import matplotlib.*")
import cma  # noqa: E402

from _base import Base  # noqa: E402


class Optimizer(Base):
    def __init__(self, space, seed=0, x0=None, sigma0=0.25, popsize=None, options=None):
        super().__init__(space, seed)
        self.x0 = None if x0 is None else self.to_unit(x0)
        self.settings = dict(options or {}, bounds=[0, 1], verbose=-9, seed=seed + 1)
        if popsize:
            self.settings["popsize"] = popsize
        self.sigma0, self.es, self.population, self.told = sigma0, None, [], []

    def propose(self):
        if self.es is None:
            x0 = self.x0 if self.x0 is not None else self.best()[0] if len(self.y) else np.full(self.dim, 0.5)
            self.es = cma.CMAEvolutionStrategy(x0, self.sigma0, self.settings)
        if self.es.stop():
            return None
        if not self.population and not self.told:
            self.population = list(self.es.ask())
        return self.population.pop(0) if self.population else None

    def observe(self, u, value):
        self.told.append((u, value))
        if len(self.told) == self.es.popsize:
            self.es.tell([t[0] for t in self.told], [t[1] for t in self.told])
            self.told = []
