"""L-BFGS-B (scipy.optimize.minimize) with forward finite-difference
gradients, in the unit cube. `continue` resumes exactly; see _local.py.

args:
  x0     start in real coordinates (default: the best point told so far, else the centre).
  fixed  {index: value} coordinates held fixed; the solver moves only the others.
  eps    finite-difference step in the unit cube (default 1e-8, scipy's default).
  ftol, gtol, maxcor   scipy options (defaults 2.2e-9, 1e-5, 10).
Each gradient costs n + 1 evaluations for n free coordinates. Unsuited to noisy objectives.
"""
from scipy.optimize import minimize

from _local import Local


class Optimizer(Local):
    def __init__(self, space, seed=0, x0=None, fixed=None, eps=1e-8, ftol=2.2e-9, gtol=1e-5, maxcor=10):
        super().__init__(space, seed, x0, fixed)
        self.options = {"eps": eps, "ftol": ftol, "gtol": gtol, "maxcor": maxcor,
                        "maxfun": 10 ** 9, "maxiter": 10 ** 9}

    def solve(self, f, x0):
        minimize(f, x0, method="L-BFGS-B", bounds=[(0, 1)] * self.ndim, options=self.options)
