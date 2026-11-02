"""Model-free batches: uniform random points and space-filling designs."""
from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from .base import check_inputs, finish


def random(X, y, q, *, pending=None, failed=None, candidates=None, dim=None, seed=0):
    """q uniform points in the cube, or q distinct candidates drawn uniformly."""
    inputs = check_inputs(X, y, q, pending, failed, candidates, dim)
    rng = np.random.default_rng(seed)
    if inputs.candidates is not None:
        return finish('random', inputs, indices=rng.choice(inputs.open, q, replace=False), settings={'seed': seed})
    return finish('random', inputs, X=rng.random((q, inputs.dim)), settings={'seed': seed})


def space_filling(X, y, q, *, pending=None, failed=None, candidates=None, dim=None, seed=0,
                  n_candidates=4096):
    """Points far from everything already placed (measured, pending or failed).

    With nothing placed in a continuous space, a scrambled Sobol sample (best
    balanced when q is a power of two). Otherwise a greedy maximin choice from
    a Sobol cloud of n_candidates points, or from the pool's candidates.
    """
    inputs = check_inputs(X, y, q, pending, failed, candidates, dim)
    known = inputs.known
    settings = {'seed': seed}
    if inputs.candidates is None and not len(known):
        points = qmc.Sobol(inputs.dim, scramble=True, seed=np.random.default_rng(seed)).random(q)
        return finish('sobol', inputs, X=points, settings=settings)
    if inputs.candidates is None:
        settings['n_candidates'] = n_candidates
        cloud = qmc.Sobol(inputs.dim, scramble=True, seed=np.random.default_rng(seed)).random(n_candidates)
    else:
        cloud = inputs.candidates
    nearest = (np.min(np.linalg.norm(cloud[:, None, :] - known[None, :, :], axis=-1), axis=1)
               if len(known) else np.full(len(cloud), np.inf))
    if inputs.candidates is not None:
        nearest[np.setdiff1d(np.arange(len(cloud)), inputs.open)] = -np.inf
    chosen = []
    for _ in range(q):
        nearest[chosen] = -np.inf
        best = int(np.argmax(nearest))
        chosen.append(best)
        nearest = np.minimum(nearest, np.linalg.norm(cloud - cloud[best], axis=1))
    info = {'min_distance_to_placed': float(np.min(np.linalg.norm(
        cloud[chosen][:, None, :] - known[None, :, :], axis=-1))) / np.sqrt(inputs.dim) if len(known) else None}
    if inputs.candidates is not None:
        return finish('space_filling', inputs, indices=chosen, settings=settings, info=info)
    return finish('space_filling', inputs, X=cloud[chosen], settings=settings, info=info)
