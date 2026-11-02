"""Gaussian-process models and batch Bayesian optimization with qLogNEI.

The model is BoTorch's SingleTaskGP with its defaults (an ARD RBF kernel with
a dimension-scaled lengthscale prior, standardized outcomes) fitted to the
unit-cube data. qlognei() proposes a whole batch with q-log noisy expected
improvement, treating pending points as already chosen, so the batch spreads
around them rather than on top of them.
"""
from __future__ import annotations

from importlib import metadata
import warnings

import numpy as np
import torch

# BoTorch's import chain warns about torch.jit; nothing here depends on it.
warnings.filterwarnings('ignore', message='.*torch.jit.script.*')

from botorch.acquisition.logei import qLogNoisyExpectedImprovement  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.models import SingleTaskGP  # noqa: E402
from botorch.models.transforms.outcome import Standardize  # noqa: E402
from botorch.optim import optimize_acqf  # noqa: E402
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402

from .base import check_inputs, finish

CHUNK = 512  # candidates scored at once in a pool


def _tensor(array):
    return torch.as_tensor(np.asarray(array, dtype=float), dtype=torch.double)


def fit_gp(X, y, seed=0):
    """Fit the GP to unit-cube X and loss y; returns (model, diagnostics)."""
    torch.manual_seed(seed)
    model = SingleTaskGP(_tensor(X), _tensor(y).unsqueeze(-1), outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model, diagnostics(model)


def diagnostics(model):
    """Lengthscales per parameter (unit-cube units; short means the loss changes fast
    along it, long means it barely matters), the noise as a share of the
    loss variance, and the leave-one-out R^2 of the fitted model."""
    kernel = model.covar_module
    lengthscale = getattr(kernel, 'lengthscale', None)
    if lengthscale is None:
        lengthscale = kernel.base_kernel.lengthscale
    model.train()
    with torch.no_grad():
        prior = model(*model.train_inputs)
        K = model.likelihood(prior).covariance_matrix
        residual = model.train_targets - prior.mean
        inverse = torch.linalg.inv(K)
        loo = (inverse @ residual) / torch.diagonal(inverse)
        targets = model.train_targets
        r2 = 1 - (loo ** 2).sum() / ((targets - targets.mean()) ** 2).sum()
    model.eval()
    return {'n': int(len(targets)), 'lengthscales': lengthscale.detach().reshape(-1).tolist(),
            'noise_share': float(model.likelihood.noise.detach().reshape(-1)[0]),
            'loo_r2': float(r2)}


def _training_data(inputs, failed_as):
    if failed_as == 'ignore' or not len(inputs.failed):
        return inputs.X, inputs.y
    if failed_as == 'worst':
        if not len(inputs.y):
            return inputs.X, inputs.y
        # A failed run counts as the worst loss seen: the model steers away from it.
        return np.vstack([inputs.X, inputs.failed]), np.r_[inputs.y, np.full(len(inputs.failed), inputs.y.max())]
    raise ValueError(f"failed_as must be 'ignore' or 'worst', not {failed_as!r}")


def _settings(seed, failed_as, num_restarts, raw_samples, bounds):
    settings = {'seed': seed, 'failed_as': failed_as, 'num_restarts': num_restarts, 'raw_samples': raw_samples,
                'model': f'SingleTaskGP defaults (botorch {metadata.version("botorch")})'}
    if bounds is not None:
        settings['bounds'] = np.asarray(bounds).tolist()
    return settings


def qlognei(X, y, q, *, pending=None, failed=None, candidates=None, dim=None, seed=0,
            bounds=None, failed_as='ignore', num_restarts=10, raw_samples=512, method='qlognei'):
    """A batch of q points (or q candidates) by q-log noisy expected improvement.

    bounds: an optional (2, d) sub-box of the unit cube to search in (rows:
    lower, upper); with candidates, only those inside it are considered.
    failed_as: 'ignore' failed runs, or count them as the 'worst' loss seen.
    """
    inputs = check_inputs(X, y, q, pending, failed, candidates, dim)
    train_X, train_y = _training_data(inputs, failed_as)
    if len(train_y) < 2:
        raise ValueError('qlognei needs at least 2 measured points; start with space_filling')
    box = np.array([[0.] * inputs.dim, [1.] * inputs.dim]) if bounds is None else np.asarray(bounds, dtype=float)
    if box.shape != (2, inputs.dim) or (box[0] > box[1]).any():
        raise ValueError(f'bounds must have shape (2, {inputs.dim}) with lower <= upper')
    # BoTorch acquisition functions maximize, so the model is fitted to -loss.
    model, info = fit_gp(train_X, -train_y, seed)
    settings = _settings(seed, failed_as, num_restarts, raw_samples, bounds)
    torch.manual_seed(seed)
    pending_points = _tensor(inputs.pending) if len(inputs.pending) else None

    if inputs.candidates is None:
        acquisition = qLogNoisyExpectedImprovement(model, X_baseline=_tensor(train_X), X_pending=pending_points)
        points, _ = optimize_acqf(acquisition, bounds=_tensor(box), q=q, num_restarts=num_restarts,
                                  raw_samples=raw_samples)
        return finish(method, inputs, X=points.detach().numpy(), settings=settings, info=info)

    inside = np.flatnonzero(((inputs.candidates >= box[0] - 1e-12) & (inputs.candidates <= box[1] + 1e-12)).all(1))
    if len(inside) < q:
        raise ValueError(f'only {len(inside)} candidates lie inside the bounds; q is {q}')
    chosen = []
    for _ in range(q):
        # Greedy batch: each pick is treated as pending for the next.
        extra = _tensor(inputs.candidates[chosen]) if chosen else None
        fantasies = extra if pending_points is None else (
            pending_points if extra is None else torch.cat([pending_points, extra]))
        acquisition = qLogNoisyExpectedImprovement(model, X_baseline=_tensor(train_X), X_pending=fantasies)
        left = np.setdiff1d(inside, chosen)
        scores = []
        with torch.no_grad():
            for start in range(0, len(left), CHUNK):
                rows = _tensor(inputs.candidates[left[start:start + CHUNK]]).unsqueeze(1)
                scores.append(acquisition(rows).reshape(-1).numpy())
        chosen.append(int(left[int(np.argmax(np.concatenate(scores)))]))
    return finish(method, inputs, indices=chosen, settings=settings, info=info)
