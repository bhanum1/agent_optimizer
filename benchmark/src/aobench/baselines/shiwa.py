"""Nevergrad's Shiwa rule-based selector, using the baseline ask/tell interface.

Uses the installed, version-pinned implementation, not a reimplementation of
the paper's decision tree. The budget, noise flag and sequential worker count
are supplied before selection. No task identity or noise-free values are used.

Nevergrad's bounded Array defaults set the initial point to the box midpoint
and mutation scales to one sixth of each range. Internal optimizer settings
are otherwise unchanged. On noisy continuous tasks Shiwa selects TBPSA.
Scoring uses the best observed point, as for every other baseline here,
rather than Nevergrad's potentially different final recommendation.
"""

from __future__ import annotations

from collections import deque
from importlib.metadata import version

import numpy as np


class Shiwa:
    """Sequential Shiwa with seeded candidates and observed-value history."""

    def __init__(self, bounds, seed=None, budget=100, noisy=False):
        import nevergrad as ng  # Optional evaluator backend; never in agent runtime.

        self.bounds = np.asarray(bounds, dtype=float)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2
                or not len(self.bounds) or not np.isfinite(self.bounds).all()
                or not (self.bounds[:, 0] < self.bounds[:, 1]).all()):
            raise ValueError("bounds must contain finite [low, high] pairs with low < high")
        self.dim = len(self.bounds)
        self.noisy = bool(noisy)
        parameter = ng.p.Array(shape=(self.dim,), lower=self.bounds[:, 0], upper=self.bounds[:, 1])
        parameter.function.deterministic = not self.noisy
        if seed is not None:
            parameter.random_state.seed(seed)
        self._shiwa = ng.optimizers.Shiwa(parametrization=parameter, budget=budget, num_workers=1)
        self._pending = deque()
        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty(0)

    def ask(self, n=1):
        """Request one point; tell its observation before requesting another."""
        if n != 1 or self._pending:
            raise ValueError("Shiwa requires sequential ask(1), tell calls")
        candidate = self._shiwa.ask()
        self._pending.append(candidate)
        return np.asarray(candidate.value, dtype=float).reshape(1, self.dim).copy()

    def tell(self, X, y):
        """Report an observation for the exact Nevergrad candidate just asked."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if (not self._pending or X.shape != (1, self.dim) or y.shape != (1,)
                or not np.isfinite(y).all()
                or not np.array_equal(X[0], self._pending[0].value)):
            raise ValueError("tell must supply the pending point and one finite observation")
        self._shiwa.tell(self._pending[0], float(y[0]))
        self._pending.popleft()
        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])

    def best(self):
        """Best evaluated point by observed value, matching other baselines."""
        if not len(self.y_):
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = int(np.argmin(self.y_))
        return self.X_[i], self.y_[i]

    def metadata(self):
        return {"nevergrad_version": version("nevergrad"),
                "selector": "Shiwa", "selected_optimizer": self._shiwa.optim.name,
                "noisy": self.noisy, "num_workers": 1,
                "parameterization": "bounded Array defaults (midpoint, range/6 scale)",
                "scoring": "best observed point"}
