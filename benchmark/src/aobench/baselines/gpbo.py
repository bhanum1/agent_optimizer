"""Standard GP Bayesian optimization (EGO-style, Jones et al. 1998), on BoTorch.

One global GP over the whole box, refit on every point, and the next point is
the maximizer of the acquisition function over the whole box. This is the
"vanilla BO" of Hvarfner et al. (2024) and BoTorch's own getting-started
recipe, with no trust region (contrast TuRBO).

Standard, no-tuning defaults:
- Initial design: 2 * dim scrambled Sobol points.
- Model: SingleTaskGP with BoTorch's default kernel and priors (RBF with
  dimension-scaled lognormal lengthscale prior), inputs normalized to the unit
  cube, outcomes standardized. Hyperparameters refit by marginal likelihood
  after every evaluation.
- Acquisition: LogExpectedImprovement (Ament et al. 2023), maximized with
  optimize_acqf (num_restarts=10, raw_samples=512).

Exact GPs cost O(n^3) per fit, so this is meant for budgets of a few hundred
evaluations.

This optimizer MINIMIZES.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from torch.quasirandom import SobolEngine

warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
warnings.filterwarnings("ignore", message=".*Optimization failed.*")  # optimize_acqf retries on its own

from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class GPBO:
    """Sequential GP-BO with LogEI. Only n=1 per `ask` after the initial design.

    Parameters
    ----------
    bounds : array-like of shape (d, 2)
    seed : int, optional
    """

    def __init__(self, bounds, seed: Optional[int] = None):
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (d, 2): one (low, high) pair per dimension")
        self.dim = self.bounds.shape[0]
        self.seed = 0 if seed is None else int(seed)
        self._lower, self._span = self.bounds[:, 0], self.bounds[:, 1] - self.bounds[:, 0]
        self._unit = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]).double()
        sobol = SobolEngine(self.dim, scramble=True, seed=self.seed)
        self._init = list(sobol.draw(2 * self.dim).double().numpy())
        self._n_asked = 0
        self.X_ = np.empty((0, self.dim))
        self.y_ = np.empty((0,))

    def ask(self, n: int = 1) -> np.ndarray:
        out = [self._next() for _ in range(n)]
        return np.array(out)

    def _next(self) -> np.ndarray:
        self._n_asked += 1
        if self._init:
            return self._lower + self._init.pop(0) * self._span
        torch.manual_seed(self.seed * 100003 + self._n_asked)
        X = torch.tensor((self.X_ - self._lower) / self._span, dtype=torch.double)
        Y = torch.tensor(self.y_, dtype=torch.double).unsqueeze(-1)
        model = SingleTaskGP(X, Y, input_transform=Normalize(d=self.dim, bounds=self._unit),
                             outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acq = LogExpectedImprovement(model, best_f=Y.min(), maximize=False)
        cand, _ = optimize_acqf(acq, bounds=self._unit, q=1, num_restarts=10, raw_samples=512)
        return self._lower + cand.detach().numpy()[0] * self._span

    def tell(self, X, y) -> None:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        self.X_ = np.vstack([self.X_, X])
        self.y_ = np.concatenate([self.y_, y])

    def best(self):
        if len(self.y_) == 0:
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = np.argmin(self.y_)
        return self.X_[i], self.y_[i]
