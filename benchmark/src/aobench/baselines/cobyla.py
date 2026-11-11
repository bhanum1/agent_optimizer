"""COBYLA (Powell, 1994): a derivative-free local solver that fits linear
models inside a trust region.

scipy's solver is closed-loop, so it runs in a background thread whose
objective callback posts each proposed point to a queue and blocks until the
main thread supplies the value.

Implementation: scipy.optimize.minimize(method="COBYLA"), which since scipy
1.16 wraps the PRIMA reference implementation. Defaults:
- The search runs in the unit cube (each coordinate rescaled by its
  bounds), with the box passed as bounds.
- Starting point: the midpoint of the box, or `x0`.
- Initial trust-region radius `rhobeg` = 0.1 in the unit cube, Py-BOBYQA's
  default for a box rescaled to the unit cube (scipy's own default of 1.0
  would span the whole cube). Final radius `tol` is left at scipy's default.
- `maxiter` is set out of reach so the external budget controls how long
  the search runs.

This optimizer MINIMIZES.
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

    Only n=1 per `ask` is supported.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Unused -- COBYLA is deterministic given a starting point.
    x0 : array-like, optional
        Starting point (default: the box midpoint).
    """

    deterministic = True  # seed is unused; one seed is enough in sweeps

    def __init__(self, bounds, seed: Optional[int] = None, x0=None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]
        self._lower = self.bounds[:, 0]
        self._span = self.bounds[:, 1] - self.bounds[:, 0]
        u0 = np.full(self.dim, 0.5) if x0 is None else (np.asarray(x0, dtype=float) - self._lower) / self._span

        self._ask_q: "queue.Queue" = queue.Queue()
        self._tell_q: "queue.Queue" = queue.Queue()
        self._last = None  # (u, y) of the most recent evaluation
        self.result = None

        def evaluate(u):
            u = np.array(u, dtype=float)
            if self._last is None or not np.array_equal(self._last[0], u):
                self._ask_q.put(self._lower + u * self._span)
                self._last = (u, self._tell_q.get())
            return self._last[1]

        def run_solver():
            self.result = minimize(
                evaluate,
                u0,
                method="COBYLA",
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

    def tell(self, X, y) -> None:
        """Record the evaluated point `X` (1, dim) and its value `y` (1,)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])
        self._tell_q.put(float(y[0]))

    def best(self):
        """Return (x_best, y_best) among all points told so far (minimum y)."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]
