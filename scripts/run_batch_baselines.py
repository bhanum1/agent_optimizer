"""Batch baselines for advisor replay: one fixed batch method per campaign, no model.

Replay's advisor proposes q points per block from scratch each time. These
baselines do the same with one method throughout, through the same
agentoptimizer.batch proposers, so a replay can be compared with batch
optimization at the same block size:

    sobol_qlognei  a Sobol first block, then qLogNEI on every later block
    sobol_turbo    a Sobol first block, then trust_region with TuRBO's length
                   rule: start at 0.8, double after 3 improving blocks in a row (at
                   most 1.6), halve after ceil(max(4, d) / q) flat ones in a row,
                   and restart at 0.8 below 1/128
    random         uniform random blocks

Indexed pool tasks and continuous tasks without constraints. Writes
<outdir>/<task>/<method>/seed_<seed>/trace.csv (eval_idx, batch_idx,
optimizer, candidate_index, y, y_best_so_far, regret) and meta.json. The first
block has --initial points (default q), so sequential runs (q = 1) start from
a design the GP can fit.

    .venv/bin/python scripts/run_batch_baselines.py --task mof_pool --budget 100 --q 8 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import time

import numpy as np

from agentoptimizer import batch
from agentoptimizer.tasks.registry import TASKS

REPO = Path(__file__).resolve().parents[1]
METHODS = ('sobol_qlognei', 'sobol_turbo', 'random')


def unit_scale(task):
    """Map raw points into the unit cube as advisor campaigns do: linear in the
    bounds, a constant pool feature getting a unit-wide range."""
    low = np.asarray(task.bounds, dtype=float)[:, 0]
    high = np.asarray(task.bounds, dtype=float)[:, 1]
    high = np.where(high > low, high, low + 1.)
    return lambda X: (np.asarray(X, dtype=float) - low) / (high - low)


def campaign(task_key, method, seed, budget, q, initial):
    task = TASKS[task_key]()
    task.reseed(seed)
    indexed = getattr(task, 'indexed_pool', False)
    if task.n_constraints or (task.pool and not indexed):
        raise ValueError('Batch baselines cover indexed pools and unconstrained continuous tasks.')
    unit = unit_scale(task)
    pool = unit(task.candidates) if indexed else None
    X, y, chosen, rows = np.empty((0, task.dim)), np.empty(0), [], []
    length, best_before, successes, failures = .8, np.inf, 0, 0
    patience = int(np.ceil(max(4, task.dim) / q))
    for block in range(1, budget + 1):
        size = min(initial if block == 1 else q, budget - len(y))
        if size <= 0:
            break
        available = np.setdiff1d(np.arange(len(pool)), chosen) if indexed else None
        inputs = {'X': X, 'y': y, 'dim': task.dim, 'seed': 1000 * seed + block,
                  'candidates': pool[available] if indexed else None}
        if method == 'random':
            proposal = batch.random(**inputs, q=size)
        elif block == 1:
            proposal = batch.space_filling(**inputs, q=size)
        elif method == 'sobol_qlognei':
            proposal = batch.qlognei(**inputs, q=size)
        else:
            proposal = batch.trust_region(**inputs, q=size, length=length)
        indices = available[proposal.indices] if indexed else None
        points = task.candidates[indices] if indexed else np.asarray(task.bounds)[:, 0] + proposal.X * (
            np.asarray(task.bounds)[:, 1] - np.asarray(task.bounds)[:, 0])
        values = [task.evaluate_index(int(i)).y if indexed else task.evaluate(p.tolist()).y
                  for i, p in zip(indices if indexed else [None] * size, points)]
        for k, value in enumerate(values):
            best = min([value, *(r['y'] for r in rows)])
            rows.append({'eval_idx': len(rows) + 1, 'batch_idx': block, 'optimizer': proposal.row_labels()[k],
                         'candidate_index': '' if not indexed else int(indices[k]), 'y': value,
                         'y_best_so_far': best, 'regret': best - task.f_opt})
        X = np.vstack([X, unit(points)])
        y = np.concatenate([y, values])
        if indexed:
            chosen.extend(int(i) for i in indices)
        if method == 'sobol_turbo' and block > 1:
            # TuRBO counts an improvement only beyond 1e-3 of the incumbent.
            if y.min() < best_before - 1e-3 * abs(best_before):
                successes, failures = successes + 1, 0
            else:
                successes, failures = 0, failures + 1
            if successes == 3:
                length, successes = min(2 * length, 1.6), 0
            if failures == patience:
                length, failures = length / 2, 0
            if length < 1 / 128:
                length = .8
        best_before = y.min()
    return task, rows


def run(job):
    task_key, method, seed, budget, q, initial, outdir = job
    start = time.time()
    task, rows = campaign(task_key, method, seed, budget, q, initial)
    folder = Path(outdir) / task_key / method / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'trace.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'meta.json').write_text(json.dumps({
        'task': task_key, 'method': method, 'seed': seed, 'budget': budget, 'q': q, 'initial': initial,
        'f_opt': float(task.f_opt), 'best_y': rows[-1]['y_best_so_far'], 'regret': rows[-1]['regret'],
        'seconds': round(time.time() - start, 1)}, indent=2) + '\n')
    return f'{task_key} {method} seed {seed}: regret {rows[-1]["regret"]:.4g} ({time.time() - start:.0f} s)'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--q', type=int, default=8)
    parser.add_argument('--initial', type=int, help='points in the first block (default q)')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--outdir', type=Path, default=REPO / 'runs/batch_baselines')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    jobs = [(args.task, m, s, args.budget, args.q, args.initial or args.q, args.outdir)
            for m in args.methods for s in args.seeds]
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(run, jobs):
            print(line, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
