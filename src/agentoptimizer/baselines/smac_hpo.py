"""SMAC (Hutter, Hoos & Leyton-Brown, 2011; SMAC3: Lindauer et al., JMLR
2022) -- Bayesian optimization with a random-forest surrogate.

This is the package's tree-surrogate baseline. A random forest models
piecewise-constant, discontinuous and categorical-like responses that a
stationary GP smooths over, so it is the standard choice for
hyperparameter optimization and mixed or conditional spaces.

Implementation: SMAC3's `HyperparameterOptimizationFacade` -- the
configuration the SMAC authors recommend for HPO -- driven through its own
ask/tell interface. Every facade component is left at its default: random
forest surrogate, log-transformed expected improvement, local search plus
random sampling of the acquisition (with SMAC's default share of
interleaved random configurations), and a scrambled Sobol initial design of
10 points per dimension capped at a quarter of the budget.

Two settings are the only departures from a stock run:
- The search space is the unit cube, one continuous hyperparameter per
  task dimension, mapped linearly to the task's bounds. Tasks expose only
  a box (integer and categorical coordinates are rounded inside the task),
  so SMAC sees the same continuous box as every other optimizer here.
- `deterministic=True`, so SMAC does not re-run a configuration under new
  seeds. Noisy tasks are therefore handled by the surrogate's own noise
  model, as for GP-BO, rather than by repeated evaluations.

The budget is needed up front because it sets the initial-design cap;
campaign.py passes it (`takes_budget = True`).

By convention, this optimizer MINIMIZES the objective (matching the rest
of this package).
"""

from __future__ import annotations

import tempfile
from typing import Optional

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue


class SMAC:
    """SMAC3 HPO facade (random-forest BO), one point per ask.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
        Lower and upper bound for each of the d dimensions.
    seed : int, optional
        Seeds SMAC's initial design, forest and acquisition search.
    budget : int
        Number of evaluations the campaign will request.
    """

    takes_budget = True  # campaign.py passes the budget to the constructor

    def __init__(self, bounds, seed: Optional[int] = None, budget: int = 100):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(
                "bounds must have shape (d, 2): one (low, high) pair per dimension"
            )
        self.dim = self.bounds.shape[0]
        self._lower = self.bounds[:, 0]
        self._span = self.bounds[:, 1] - self.bounds[:, 0]
        self._names = [f"x{i:03d}" for i in range(self.dim)]

        space = ConfigurationSpace(seed=0 if seed is None else seed)
        space.add([Float(name, (0.0, 1.0)) for name in self._names])
        self._outdir = tempfile.TemporaryDirectory(prefix="smac_")
        scenario = Scenario(
            space,
            deterministic=True,
            n_trials=int(budget),
            seed=0 if seed is None else seed,
            output_directory=self._outdir.name,
        )
        self._smac = HyperparameterOptimizationFacade(
            scenario,
            target_function=lambda config, seed=0: 0.0,  # unused: ask/tell only
            overwrite=True,
            logging_level=40,
        )
        self._pending: list = []

        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        """Propose `n` candidate points, shape (n, dim)."""
        infos = [self._smac.ask() for _ in range(n)]
        self._pending.extend(infos)
        U = np.array([[info.config[name] for name in self._names] for info in infos])
        return self._lower + U * self._span

    def tell(self, X, y) -> None:
        """Record values `y` for the points `X`, in the order they were asked."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        for yi in y:
            self._smac.tell(self._pending.pop(0), TrialValue(cost=float(yi)), save=False)
        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])

    def best(self):
        """Return (x_best, y_best) among all points told so far (minimum y)."""
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]
