"""GP helpers for gp_bo and turbo (botorch SingleTaskGP, fitted by marginal likelihood)."""
import warnings

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)


def model_values(y):
    """Values the GP models: infeasible merits (>= 1e12) are capped at the worst
    feasible value plus its range, so penalties do not swamp the fit."""
    y = np.asarray(y, dtype=float)
    feasible = y < 1e12
    if feasible.any() and not feasible.all():
        lo, hi = y[feasible].min(), y[feasible].max()
        y = np.where(feasible, y, hi + (hi - lo or 1.0))
    return y


def fit(U, targets, seed):
    """GP on `targets` (to be maximized; pass -model_values(y))."""
    torch.manual_seed(seed)
    X = torch.tensor(np.asarray(U))
    Y = torch.tensor(np.asarray(targets, dtype=float)).unsqueeze(-1)
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model
