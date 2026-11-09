"""Powell's conjugate-direction method (scipy.optimize.minimize), derivative-free,
in the unit cube. `continue` resumes exactly; see _local.py.

args:
  x0     start in real coordinates (default: the best point told so far, else the centre).
  fixed  {index: value} coordinates held fixed; the solver moves only the others.
  xtol, ftol   scipy options (defaults 1e-4, 1e-4).
"""
from scipy.optimize import minimize

from _local import Local


class Optimizer(Local):
    def __init__(self, space, seed=0, x0=None, fixed=None, xtol=1e-4, ftol=1e-4):
        super().__init__(space, seed, x0, fixed)
        self.options = {"xtol": xtol, "ftol": ftol, "maxfev": 10 ** 9, "maxiter": 10 ** 9}

    def solve(self, f, x0):
        minimize(f, x0, method="Powell", bounds=[(0, 1)] * self.ndim, options=self.options)
