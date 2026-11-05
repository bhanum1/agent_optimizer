"""COBYLA (scipy.optimize.minimize): derivative-free linear-approximation
method that handles the black-box inequality constraints directly (not
through a penalty), in the unit cube. `continue` resumes exactly; see _local.py.

args:
  x0      start in real coordinates (default: the best point told so far, else the centre).
  fixed   {index: value} coordinates held fixed; the solver moves only the others.
  rhobeg  initial step in the unit cube (default 0.1).
  tol     final trust radius (default 1e-6).
Each evaluation gives the objective and all constraints at once.
"""
import numpy as np
from scipy.optimize import minimize

from _local import Local


class Optimizer(Local):
    def __init__(self, space, seed=0, x0=None, fixed=None, rhobeg=0.1, tol=1e-6):
        super().__init__(space, seed, x0, fixed)
        self.rhobeg, self.tol = rhobeg, tol

    def solve(self, f, x0):
        cache = {}

        def evaluate(u):
            key = tuple(np.round(u, 15))
            if key not in cache:
                f(u)  # value is reported back through observe; raw y, g via self.last
                cache[key] = self.last
            return cache[key]

        constraints = [{"type": "ineq", "fun": (lambda u, i=i: -evaluate(u)[1][i])}
                       for i in range(self.space["n_constraints"])]
        minimize(lambda u: evaluate(u)[0], x0, method="COBYLA", bounds=[(0, 1)] * self.ndim,
                 constraints=constraints, options={"rhobeg": self.rhobeg, "tol": self.tol, "maxiter": 10 ** 9})
