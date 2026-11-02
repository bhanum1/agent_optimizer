"""Simulated lab for the reaction example: measures every pending experiment.

    .venv/bin/python examples/reaction/lab.py <campaign folder>

Plays the bench chemist. Each pending row gets a yield from a hidden
response surface plus measurement noise; very hot, long runs fail. The
results go to blocks/NNN/lab_results.csv and into data.csv through
opt_run.py's record. This file is the hidden truth: the advisor must not
read it.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import numpy as np

from agentoptimizer.advisor.data import load_campaign
from agentoptimizer.advisor.results import record_results

NOISE_SD = 1.5  # yield points


def true_yield(temperature, catalyst, base_equiv, hours):
    """Yield in %, best (about 95) near 85 C, 1.2 mol%, 2.4 equiv, 10 h.
    Hotter runs want shorter times (the t*h term)."""
    t = (temperature - 85) / 30
    c = (math.log10(catalyst) - math.log10(1.2)) / .6
    b = (base_equiv - 2.4) / 1.2
    h = (hours - 10) / 8
    return 3 + 92 * math.exp(-(t * t + c * c + b * b + .5 * h * h + .8 * t * h))


def decomposes(temperature, hours):
    return temperature > 110 and hours > 16


def measure(campaign):
    """Record a result for every pending row; returns the results file path, or None."""
    problem, _, data = load_campaign(campaign)
    pending = np.flatnonzero(data.where('pending'))
    if not len(pending):
        return None
    rows = []
    for i in pending:
        identifier = int(data.ids[i])
        values = dict(zip(problem.names, data.X[i]))
        if decomposes(values['temperature'], values['hours']):
            rows.append([identifier, '', 'failed', 'black tar, no product peak'])
            continue
        noise = np.random.default_rng(identifier).normal(0, NOISE_SD)
        rows.append([identifier, f'{min(100., max(0., true_yield(*data.X[i]) + noise)):.1f}', '', ''])
    folder = Path(campaign) / 'blocks' / f'{int(data.blocks[pending].max()):03d}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'lab_results.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['id', problem.objective, 'status', 'notes'])
        writer.writerows(rows)
    record_results(campaign, path)
    return path


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    result = measure(sys.argv[1])
    print(f'Measured and recorded: {result}' if result else 'Nothing pending.')
