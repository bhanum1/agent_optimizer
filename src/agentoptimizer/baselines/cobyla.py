"""COBYLA (Powell, 1994) -- a linear-model trust-region local solver that
handles inequality constraints directly.

This is the package's constraint-handling baseline. Every other optimizer
here sees only the penalized merit computed in baselines/campaign.py;
COBYLA is told the raw objective and the raw constraint values instead
(`uses_constraints = True`), since its whole method is to model the
constraints separately from the objective. On an unconstrained task it is
simply a derivative-free local solver with linear models.

COBYLA is closed-loop, like BOBYQA, and is driven the same way: scipy's
solver runs in a background thread whose callbacks post each proposed point
to a queue and block until the main thread supplies the value. scipy calls
the objective and then the constraints at the same point, so the
constraint callback reuses the values told for that point.

Implementation: scipy.optimize.minimize(method="COBYLA"), which since scipy
1.16 wraps the PRIMA reference implementation.

Standard, no-tuning defaults, matching BOBYQA's conventions:
- The search runs in the unit cube (each coordinate rescaled by its
  bounds), with the box passed as bounds.
- Starting point: the midpoint of the box.
- Initial trust-region radius `rhobeg` = 0.1 in the unit cube, the same
  value Py-BOBYQA uses by default for a box rescaled to the unit cube
  (scipy's own default of 1.0 assumes no scaling and would span the
  whole cube). Final radius `tol` is left at scipy's default.
- `maxiter` is set out of reach so the external budget, not the solver's
  own cap (1000 * dim), controls how long the search runs.

By convention, this optimizer MINIMIZES the objective (matching the rest
of this package). Constraints follow the task convention: feasible iff
every g(x) <= 0.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from . import OptimizerExhausted


class COBYLA:
    """COBYLA, driven one point at a time via a background solver thread.

    Only n=1 per `ask` is supported. `tell` takes the raw objective value
    and, for a constrained task, the constraint values at that point.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Unused -- COBYLA is deterministic given a starting point.
    n_constraints : int
        Number of inequality constraints the task reports (0 if none).

    Example
    -------
        opt = COBYLA(bounds=[(-5, 5)] * 10, n_constraints=2)
        X = opt.ask(n=1)
        ev = task.evaluate(X[0])
        opt.tell(X, [ev.y], G=[ev.g])
    """

    deterministic = True  # seed is unused; one seed is enough in sweeps
    uses_constraints = True  # campaign.py passes raw y and g, not the merit

    def __init__(self, bounds, seed: Optional[int] = None, n_constraints: int = 0):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]
        self.n_constraints = int(n_constraints)
        self._lower = self.bounds[:, 0]
        self._span = self.bounds[:, 1] - self.bounds[:, 0]

        self._ask_q: "queue.Queue" = queue.Queue()
        self._tell_q: "queue.Queue" = queue.Queue()
        self._last = None  # (u, y, g) of the most recent evaluation
        self.result = None

        def evaluate(u):
            u = np.array(u, dtype=float)
            if self._last is None or not np.array_equal(self._last[0], u):
                self._ask_q.put(self._lower + u * self._span)
                y, g = self._tell_q.get()
                self._last = (u, y, g)
            return self._last

        def objective(u):
            return evaluate(u)[1]

        def constraints(u):
            # scipy's convention is feasible iff c(u) >= 0; the task's is g <= 0.
            return -evaluate(u)[2]

        cons = ([{"type": "ineq", "fun": constraints}] if self.n_constraints else [])

        def run_solver():
            self.result = minimize(
                objective,
                np.full(self.dim, 0.5),
                method="COBYLA",
                constraints=cons,
                bounds=[(0.0, 1.0)] * self.dim,
                options={"rhobeg": 0.1, "maxiter": 10**9},
            )
            self._ask_q.put(None)  # sentinel: the solver thread has returned

        self._thread = threading.Thread(target=run_solver, daemon=True)
        self._thread.start()

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose 1 new candidate point, shape (1, dim)."""
        if n != 1:
            raise NotImplementedError(
                "COBYLA only supports asking for one point at a time (n=1); "
                "it is a strictly sequential local solver."
            )
        x = self._ask_q.get()
        if x is None:
            raise OptimizerExhausted(
                "COBYLA's solver thread has already returned (converged, or "
                "hit a stopping condition) and will not propose further "
                f"points. Final result: {self.result}"
            )
        return x.reshape(1, -1)

    def tell(self, X, y, G=None) -> None:
        """Record the evaluated point `X` (1, dim), its raw objective `y` (1,)
        and, for a constrained task, its constraint values `G` (1, m)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        g = (np.asarray(G[0], dtype=float).reshape(-1) if G is not None
             else np.empty(0))
        if len(g) != self.n_constraints:
            raise ValueError(f"expected {self.n_constraints} constraint values, got {len(g)}")
        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])
        self._tell_q.put((float(y[0]), g))

    def best(self):
        """Return (x_best, y_best) among all points told so far (minimum y,
        ignoring feasibility; campaign.py tracks the best feasible value)."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]
