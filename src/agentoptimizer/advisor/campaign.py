"""What a proposal script sees of its campaign.

    from agentoptimizer.advisor import Campaign

    campaign = Campaign.load(__file__)      # from blocks/NNN/propose.py
    X, y = campaign.observations()          # done rows: unit cube, y to minimize
    ...                                     # choose points U in [0, 1]^dim
    campaign.write_proposals(U, label='gp_bo')

With a [pool], choose rows instead: candidates(), available() and
write_candidates(positions, label).

The optimizers in agentoptimizer.optimizers minimize over a box, so the
default view is the unit cube with the objective as a loss: negated when the
problem maximizes. raw=True gives the user's units and direction instead.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .data import format_number, load_campaign
from .problem import ProblemError

# Proposed values are rounded to this many significant digits, so a lab
# protocol does not read 140.00000000000003.
DIGITS = 6


class Campaign:
    def __init__(self, root, block=None):
        self.root = Path(root)
        self.problem, self.table, self.data = load_campaign(self.root)
        self.block = block

    @classmethod
    def load(cls, path='.'):
        """The campaign containing `path` (a folder, or a file such as __file__)."""
        path = Path(path).resolve()
        start = path if path.is_dir() else path.parent
        for folder in [start, *start.parents]:
            if (folder / 'problem.toml').exists():
                parts = start.relative_to(folder).parts
                block = int(parts[1]) if len(parts) >= 2 and parts[0] == 'blocks' and parts[1].isdigit() else None
                return cls(folder, block)
        raise ProblemError(f'no problem.toml in {start} or any folder above it')

    @property
    def dim(self):
        return self.problem.dim

    @property
    def names(self):
        return self.problem.names

    @property
    def bounds(self):
        """Raw bounds, shape (dim, 2)."""
        return self.problem.bounds

    @property
    def unit_bounds(self):
        return np.array([[0., 1.]] * self.dim)

    @property
    def remaining(self):
        """Experiments left in the budget after done, failed and pending ones; None without a total."""
        if self.problem.total_budget is None:
            return None
        return self.problem.total_budget - len(self.data.ids)

    def _points(self, status, raw):
        X = self.data.X[self.data.where(status)]
        return X.copy() if raw else self.problem.to_unit(X)

    def observations(self, raw=False):
        """Done rows as (X, y): unit cube and loss by default, else raw units and the user's direction."""
        y = self.data.y[self.data.where('done')]
        return self._points('done', raw), (y.copy() if raw else self.problem.sign * y)

    def pending(self, raw=False):
        """Points proposed but not yet measured; propose around them, not on top of them."""
        return self._points('pending', raw)

    def failed(self, raw=False):
        """Points whose experiment failed, with no result."""
        return self._points('failed', raw)

    def incumbent(self, raw=False):
        """The best done point and its value, in the same convention as observations()."""
        X, y = self.observations(raw)
        if not len(y):
            return None, None
        best = np.argmax(y) if raw and self.problem.direction == 'maximize' else np.argmin(y)
        return X[best], y[best]

    @property
    def candidate_ids(self):
        """Pool mode: the candidate_id of each pool row, in file order."""
        return list(self._pool().ids)

    def candidates(self, raw=False):
        """Pool mode: every candidate's features, one row per pool row."""
        X = self._pool().X
        return X.copy() if raw else self.problem.to_unit(X)

    def available(self):
        """Pool mode: positions of the pool rows not yet done or pending."""
        taken = set(self.data.candidate[self.data.where('done') | self.data.where('pending')].tolist())
        return np.array([i for i, c in enumerate(self._pool().ids) if c not in taken], dtype=int)

    def _pool(self):
        if self.problem.pool is None:
            raise ProblemError('this campaign has no [pool]; propose points with write_proposals()')
        return self.problem.pool

    def write_candidates(self, positions, label, notes=None):
        """Pool mode: write proposals.csv naming the pool rows at `positions`. Returns the path."""
        pool = self._pool()
        positions = [int(i) for i in np.atleast_1d(positions)]
        notes = notes if notes is not None else [''] * len(positions)
        path = self._proposals_path()
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['candidate_id', 'optimizer', 'notes'])
            for i, note in zip(positions, notes):
                writer.writerow([pool.ids[i], label, note])
        print(f'Wrote {len(positions)} proposals to {path}')
        return path

    def _proposals_path(self):
        if self.block is None:
            raise ProblemError('proposals are written from a script inside blocks/NNN/')
        return self.root / 'blocks' / f'{self.block:03d}' / 'proposals.csv'

    def write_proposals(self, X, label, raw=False, notes=None):
        """Write blocks/NNN/proposals.csv: one row per point, rounded to DIGITS significant
        digits (integers to whole numbers) and kept in bounds. Returns the path."""
        if self.problem.pool is not None:
            raise ProblemError('this campaign has a [pool]; propose rows with write_candidates()')
        path = self._proposals_path()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.dim:
            raise ValueError(f'expected points with {self.dim} coordinates, got shape {X.shape}')
        if not raw:
            X = self.problem.from_unit(X)
        X = np.array([[float(f'{v:.{DIGITS}g}') for v in row] for row in X]).reshape(X.shape)
        X = np.clip(X, self.bounds[:, 0], self.bounds[:, 1])
        notes = notes if notes is not None else [''] * len(X)
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow([*self.names, 'optimizer', 'notes'])
            for row, note in zip(X, notes):
                writer.writerow([*(format_number(p, v) for p, v in zip(self.problem.parameters, row)), label, note])
        print(f'Wrote {len(X)} proposals to {path}')
        return path
