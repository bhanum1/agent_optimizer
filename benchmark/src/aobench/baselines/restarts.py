"""Independent restarts from uniform random points.

`Restarting(cls, bounds, seed)` starts `cls` at a uniform random point in the
box. Whenever the solver signals OptimizerExhausted (it converged or stopped),
a fresh instance with the same settings starts at a new uniform random point.
So the external budget is always used up, as in COCO practice. Nothing is
carried over between restarts: no step size, no population size (CMA-ES is
plain restarts, not IPOP/BIPOP).

The start points come from their own generator, seeded by `seed`, so a
campaign is reproducible. Restart k's solver gets seed `seed + 100_003 * k`.
"""

from __future__ import annotations

import numpy as np

from . import OptimizerExhausted

MAX_RESTARTS = 100_000  # a guard against a solver that exhausts at once, forever


class Restarting:
    def __init__(self, cls, bounds, seed: int, **kwargs):
        self.cls, self.kwargs = cls, kwargs
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = self.bounds.shape[0]
        self.seed = int(seed)
        self._starts = np.random.default_rng([self.seed, 20261110])
        self.n_restarts = 0  # restarts after the first start
        self._inner = self._new(0)

    def _new(self, k: int):
        x0 = self._starts.uniform(self.bounds[:, 0], self.bounds[:, 1])
        return self.cls(bounds=self.bounds, seed=self.seed + 100_003 * k, x0=x0, **self.kwargs)

    def ask(self, n: int = 1) -> np.ndarray:
        while True:
            try:
                return self._inner.ask(n)
            except OptimizerExhausted:
                self.n_restarts += 1
                if self.n_restarts > MAX_RESTARTS:
                    raise
                self._inner = self._new(self.n_restarts)

    def tell(self, X, y, **kwargs) -> None:
        self._inner.tell(X, y, **kwargs)
