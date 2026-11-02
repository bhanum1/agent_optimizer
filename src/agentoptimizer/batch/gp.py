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
            bounds=None, failed_as='ignore', num_restarts=10, raw_samples=512):
    """A batch of q points (or q candidates) by q-log noisy expected improvement.

    bounds: an optional (2, d) sub-box of the unit cube to search in (rows:
    lower, upper); with candidates, only those inside it are considered.
    failed_as: 'ignore' failed runs, or count them as the 'worst' loss seen.
    """
    inputs = check_inputs(X, y, q, pending, failed, candidates, dim)
    model, info, train_X = _model(inputs, failed_as, seed)
    box = np.array([[0.] * inputs.dim, [1.] * inputs.dim]) if bounds is None else np.asarray(bounds, dtype=float)
    if box.shape != (2, inputs.dim) or (box[0] > box[1]).any():
        raise ValueError(f'bounds must have shape (2, {inputs.dim}) with lower <= upper')
    settings = _settings(seed, failed_as, num_restarts, raw_samples, bounds)
    return _propose('qlognei', inputs, model, info, train_X, box, q, seed, settings, num_restarts, raw_samples)


def _model(inputs, failed_as, seed):
    train_X, train_y = _training_data(inputs, failed_as)
    if len(train_y) < 2:
        raise ValueError('a GP proposer needs at least 2 measured points; start with space_filling')
    # BoTorch acquisition functions maximize, so the model is fitted to -loss.
    model, info = fit_gp(train_X, -train_y, seed)
    return model, info, train_X


def _inside(candidates, box):
    return np.flatnonzero(((candidates >= box[0] - 1e-12) & (candidates <= box[1] + 1e-12)).all(1))


def _propose(method, inputs, model, info, train_X, box, q, seed, settings, num_restarts, raw_samples):
    """qLogNEI over the box: optimized jointly for points, greedily for candidates."""
    torch.manual_seed(seed)
    pending_points = _tensor(inputs.pending) if len(inputs.pending) else None
    if inputs.candidates is None:
        acquisition = qLogNoisyExpectedImprovement(model, X_baseline=_tensor(train_X), X_pending=pending_points)
        points, _ = optimize_acqf(acquisition, bounds=_tensor(box), q=q, num_restarts=num_restarts,
                                  raw_samples=raw_samples)
        return finish(method, inputs, X=points.detach().numpy(), settings=settings, info=info)

    inside = _inside(inputs.candidates, box)
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


def trust_region(X, y, q, *, pending=None, failed=None, candidates=None, dim=None, seed=0,
                 length=0.4, center=None, failed_as='ignore', num_restarts=10, raw_samples=512):
    """qLogNEI inside a box around the incumbent, as in TuRBO.

    length: the box edge in the unit cube before scaling. TuRBO starts at 0.8,
    doubles it (to at most 1.6) after a run of improving batches, halves it
    after a run of non-improving ones, and restarts below 2**-7; choose it
    from the block history. Edges are stretched along parameters with long
    lengthscales and shrunk along short ones, keeping the volume length**d.
    center: a unit-cube point; by default the incumbent (lowest loss).
    With candidates, the box grows until it holds at least q of them.
    """
    inputs = check_inputs(X, y, q, pending, failed, candidates, dim)
    if not 0 < length <= 2:
        raise ValueError(f'length must be in (0, 2], not {length!r}')
    model, info, train_X = _model(inputs, failed_as, seed)
    scales = np.asarray(info['lengthscales'])
    weights = scales / np.exp(np.mean(np.log(scales)))
    middle = inputs.X[np.argmin(inputs.y)] if center is None else _check_center(center, inputs.dim)
    used = length
    while True:
        box = np.clip(np.array([middle - used * weights / 2, middle + used * weights / 2]), 0., 1.)
        if inputs.candidates is None or len(_inside(inputs.candidates, box)) >= q or (box == [[0], [1]]).all():
            break
        used *= 1.25
    settings = {**_settings(seed, failed_as, num_restarts, raw_samples, None), 'length': length,
                'center': 'incumbent' if center is None else middle.tolist()}
    info = {**info, 'center': middle.tolist(), 'length_used': used, 'box': box.tolist()}
    return _propose('trust_region', inputs, model, info, train_X, box, q, seed, settings, num_restarts, raw_samples)


def _check_center(center, dim):
    center = np.asarray(center, dtype=float).reshape(-1)
    if center.shape != (dim,) or center.min() < 0 or center.max() > 1:
        raise ValueError(f'center must be a point in the unit cube with {dim} coordinates')
    return center
