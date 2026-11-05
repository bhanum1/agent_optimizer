"""BOBYQA (Py-BOBYQA): derivative-free trust-region method with quadratic
models, for bound-constrained problems, in the unit cube. `continue` resumes
exactly; see _local.py.

args:
  x0      start in real coordinates (default: the best point told so far, else the centre).
  fixed   {index: value} coordinates held fixed; the solver moves only the others.
  rhobeg  initial trust radius in the unit cube (default 0.1).
  rhoend  final trust radius (default 1e-8).
  npt     interpolation points (default 2 * n + 1 for n free coordinates; at least n + 2).
Building the first model of a run costs npt evaluations.
"""
import numpy as np
import pybobyqa

from _local import Local


class Optimizer(Local):
    def __init__(self, space, seed=0, x0=None, fixed=None, rhobeg=0.1, rhoend=1e-8, npt=None):
        super().__init__(space, seed, x0, fixed)
        self.rhobeg, self.rhoend = rhobeg, rhoend
        self.npt = npt or 2 * self.ndim + 1

    def solve(self, f, x0):
        # Py-BOBYQA needs x0 at least rhobeg from each bound it may move toward.
        pybobyqa.solve(f, x0, bounds=(np.zeros(self.ndim), np.ones(self.ndim)), npt=self.npt,
                       # Py-BOBYQA keeps a list of length maxfun and inserts at its front every
                       # step, so 10**9 cost ~8 GB and ~1 s per step. 10**5 exceeds any budget.
                       rhobeg=self.rhobeg, rhoend=self.rhoend, maxfun=10 ** 5,
                       seek_global_minimum=False, do_logging=False)
