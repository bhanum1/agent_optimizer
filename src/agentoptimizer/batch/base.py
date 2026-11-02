"""What every batch proposer shares: inputs, the Proposal it returns, pool snapping.

A proposer is a function

    proposer(X, y, q, *, pending=None, failed=None, candidates=None, dim=None,
             seed=0, **settings) -> Proposal

X (n, d) are measured points in the unit cube and y (n,) a loss to minimize.
pending are points proposed but not measured yet; failed are points whose
experiment gave no value. With candidates (rows a pool still allows, in the
unit cube), the proposal picks q of them. dim is needed only when nothing has
been measured and there is no pool. The same inputs and seed give the same
Proposal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Proposal:
    method: str
    X: np.ndarray  # (q, d) points in the unit cube
    indices: np.ndarray | None = None  # with candidates: the chosen rows' positions
    settings: dict = field(default_factory=dict)  # every setting used, defaults included
    info: dict = field(default_factory=dict)  # diagnostics: model fit, trust region, ...
    labels: list | None = None  # per-row method labels, for combined proposals

    def row_labels(self):
        return list(self.labels) if self.labels is not None else [self.method] * len(self.X)

    def describe(self):
        def text(value):
            if isinstance(value, float):
                return f'{value:.4g}'
            if isinstance(value, (list, tuple, np.ndarray)):
                return '[' + ', '.join(text(v) for v in np.asarray(value).ravel().tolist()) + ']'
            return str(value)
        lines = [f'{self.method}: {len(self.X)} points']
        lines += [f'  setting {k} = {text(v)}' for k, v in self.settings.items()]
        lines += [f'  info {k} = {text(v)}' for k, v in self.info.items()]
        return '\n'.join(lines)


@dataclass
class Inputs:
    X: np.ndarray
    y: np.ndarray
    pending: np.ndarray
    failed: np.ndarray
    candidates: np.ndarray | None
    dim: int

    @property
    def known(self):
        """Every point already placed: measured, pending or failed."""
        return np.vstack([self.X, self.pending, self.failed])

    @property
    def open(self):
        """Positions of the candidates not already pending: a combined block passes
        its earlier parts as pending, and they must not be chosen twice."""
        if self.candidates is None:
            return None
        if not len(self.pending):
            return np.arange(len(self.candidates))
        taken = np.isclose(self.candidates[:, None, :], self.pending[None, :, :], atol=1e-12).all(-1).any(1)
        return np.flatnonzero(~taken)


def _points(name, value, dim):
    if value is None:
        return np.empty((0, dim))
    array = np.atleast_2d(np.asarray(value, dtype=float))
    if array.size == 0:
        return np.empty((0, dim))
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f'{name} must have shape (n, {dim}), not {array.shape}')
    if not np.isfinite(array).all() or array.min() < -1e-9 or array.max() > 1 + 1e-9:
        raise ValueError(f'{name} must be finite and inside the unit cube')
    return np.clip(array, 0., 1.)


def check_inputs(X, y, q, pending=None, failed=None, candidates=None, dim=None):
    """Validate a proposer's inputs and infer the dimension."""
    for value in (X, pending, failed, candidates):
        if dim is None and value is not None and np.asarray(value).size:
            dim = np.atleast_2d(np.asarray(value)).shape[1]
    if dim is None:
        raise ValueError('nothing measured and no candidates: pass dim')
    X = _points('X', X, dim)
    y = np.asarray(y if y is not None else [], dtype=float).reshape(-1)
    if len(y) != len(X):
        raise ValueError(f'X has {len(X)} rows but y has {len(y)} values')
    if not np.isfinite(y).all():
        raise ValueError('y must be finite; leave failed experiments out of X and y')
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 1:
        raise ValueError(f'q must be a positive integer, not {q!r}')
    pool = None if candidates is None else _points('candidates', candidates, dim)
    inputs = Inputs(X, y, _points('pending', pending, dim), _points('failed', failed, dim), pool, dim)
    if pool is not None and q > len(inputs.open):
        raise ValueError(f'q = {q} exceeds the {len(inputs.open)} candidates available')
    return inputs


def snap(points, candidates, allowed=None):
    """Positions of distinct candidates nearest to each point, in order,
    among the `allowed` positions (default: all)."""
    distances = np.linalg.norm(points[:, None, :] - candidates[None, :, :], axis=-1)
    if allowed is not None:
        distances[:, np.setdiff1d(np.arange(len(candidates)), allowed)] = np.inf
    chosen = []
    for row in distances:
        row = row.copy()
        row[chosen] = np.inf
        chosen.append(int(np.argmin(row)))
    return np.array(chosen, dtype=int)


def combine(*proposals):
    """One block from several proposals, in order; each row keeps its method label.
    Give each later proposer the earlier parts as pending, so they are not repeated:
    pending=np.vstack([inputs['pending'], first.X])."""
    if not proposals:
        raise ValueError('combine needs at least one proposal')
    pool = proposals[0].indices is not None
    if any((p.indices is not None) != pool for p in proposals):
        raise ValueError('cannot combine pool rows with free points')
    indices = np.concatenate([p.indices for p in proposals]) if pool else None
    if pool and len(set(indices.tolist())) < len(indices):
        raise ValueError('the proposals pick the same candidate twice; pass earlier parts as pending')
    methods = [p.method for p in proposals]
    return Proposal('+'.join(dict.fromkeys(methods)), np.vstack([p.X for p in proposals]), indices,
                    {f'{i}:{p.method}': p.settings for i, p in enumerate(proposals, 1)},
                    {f'{i}:{p.method}': p.info for i, p in enumerate(proposals, 1)},
                    [label for p in proposals for label in p.row_labels()])


def finish(method, inputs, X=None, indices=None, settings=None, info=None):
    """Build the Proposal; with candidates, X is taken from the chosen rows."""
    if inputs.candidates is not None:
        indices = np.asarray(indices, dtype=int)
        X = inputs.candidates[indices]
    return Proposal(method, np.clip(np.asarray(X, dtype=float), 0., 1.), indices, settings or {}, info or {})
