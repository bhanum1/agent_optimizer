"""What a proposal script sees of its campaign.

blocks/NNN/propose.py sets the block's SIZE and defines propose(campaign, q),
which returns the next q points as an agentoptimizer.batch Proposal. The
executor calls it once per round, on the data so far:

    from agentoptimizer import batch

    SIZE = 12

    def propose(campaign, q):
        return batch.qlognei(**campaign.inputs(), q=q, seed=100 * campaign.block + campaign.round)

The proposers in agentoptimizer.batch take inputs(): the data in the unit
cube with the objective as a loss (negated when the problem maximizes) and,
with a [pool], the available rows as candidates. For points of your own,
return batch.Proposal(label, U) with U in the unit cube, or for a pool
batch.Proposal(label, candidates[positions], indices=positions), with
positions into inputs()['candidates']. observations(), incumbent() and
friends give the same data directly; raw=True gives the user's units and
direction instead.
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


def _labels(label, count):
    """One method label per row: a single label, or one per row."""
    labels = [label] * count if isinstance(label, str) else list(label)
    if len(labels) != count or not all(isinstance(text, str) and text for text in labels):
        raise ValueError(f'expected one label or {count} non-empty labels')
    return labels


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
    def round(self):
        """The round being proposed within this block, from 1; derive seeds from it."""
        return self.data.in_block(self.block) // self.problem.round_size + 1

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

    def inputs(self):
        """Keyword arguments for an agentoptimizer.batch proposer: the data so far
        in the unit cube as a loss and, with a pool, the available rows as candidates."""
        X, y = self.observations()
        pool = self.problem.pool is not None
        return {'X': X, 'y': y, 'pending': self.pending(), 'failed': self.failed(), 'dim': self.dim,
                'candidates': self.candidates()[self.available()] if pool else None}

    def write(self, proposal, notes=None):
        """Write the round's proposals.csv from a Proposal made from inputs(); prints its
        settings and diagnostics, which `opt_run.py run` keeps in propose.log. Returns the path."""
        print(proposal.describe())
        if self.problem.pool is not None:
            if proposal.indices is None:
                raise ProblemError('this campaign has a [pool]; pass candidates=inputs()["candidates"]')
            return self._write_candidates(self.available()[proposal.indices], proposal.row_labels(), notes)
        return self._write_points(proposal.X, proposal.row_labels(), notes=notes)

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
            raise ProblemError('this campaign has no [pool]')
        return self.problem.pool

    def _write_candidates(self, positions, label, notes=None):
        """Pool mode: write proposals.csv naming the pool rows at `positions`. Returns the path."""
        pool = self._pool()
        positions = [int(i) for i in np.atleast_1d(positions)]
        notes = notes if notes is not None else [''] * len(positions)
        path = self._proposals_path()
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['candidate_id', 'optimizer', 'notes'])
            for i, row_label, note in zip(positions, _labels(label, len(positions)), notes):
                writer.writerow([pool.ids[i], row_label, note])
        print(f'Wrote {len(positions)} proposals to {path}')
        return path

    def _proposals_path(self):
        if self.block is None:
            raise ProblemError('proposals are written from a script inside blocks/NNN/')
        return self.root / 'blocks' / f'{self.block:03d}' / 'proposals.csv'

    def _write_points(self, X, label, notes=None):
        """Write blocks/NNN/proposals.csv: one row per point, rounded to DIGITS significant
        digits (integers to whole numbers) and kept in bounds. Returns the path."""
        if self.problem.pool is not None:
            raise ProblemError('this campaign has a [pool]; propose rows of it')
        path = self._proposals_path()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.dim:
            raise ValueError(f'expected points with {self.dim} coordinates, got shape {X.shape}')
        X = self.problem.from_unit(X)
        X = np.array([[float(f'{v:.{DIGITS}g}') for v in row] for row in X]).reshape(X.shape)
        X = np.clip(X, self.bounds[:, 0], self.bounds[:, 1])
        notes = notes if notes is not None else [''] * len(X)
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow([*self.names, 'optimizer', 'notes'])
            for row, row_label, note in zip(X, _labels(label, len(X)), notes):
                writer.writerow([*(format_number(p, v) for p, v in zip(self.problem.parameters, row)), row_label, note])
        print(f'Wrote {len(X)} proposals to {path}')
        return path
