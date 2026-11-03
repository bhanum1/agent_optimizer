"""A fixed summary of a campaign's data, for the advisor to read.

Computed here rather than by the model, so the numbers are exact, and saved
with each block, so its decision can be traced to what it was shown. Values
are in the user's units and direction. Distances are taken in the unit cube
and divided by sqrt(dim), so they lie in [0, 1].
"""
from __future__ import annotations

import math

import numpy as np

# Unit-cube distance treated as touching a bound, or as the same point.
TOUCH = 1e-3
TOP = 5


def _number(value):
    return None if value is None or not math.isfinite(value) else float(value)


def _distances(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def _at_bounds(problem, x):
    sides = []
    for i, (p, value, unit) in enumerate(zip(problem.parameters, x, problem.to_unit([x])[0])):
        if problem.pool is not None and np.ptp(problem.pool.X[:, i]) == 0:
            continue  # a constant pool feature says nothing about where to search
        low = value == p.low if p.type == 'integer' else unit < TOUCH
        high = value == p.high if p.type == 'integer' else unit > 1 - TOUCH
        sides += [{'parameter': p.name, 'side': side} for side, hit in (('low', low), ('high', high)) if hit]
    return sides


def _row(problem, data, index):
    row = {'id': int(data.ids[index]), 'block': int(data.blocks[index]), 'value': _number(data.y[index]),
           'parameters': {name: float(v) for name, v in zip(problem.names, data.X[index])}}
    if problem.pool is not None:
        row['candidate_id'] = data.candidate[index]
    return row


def summarize(problem, data):
    done, pending, failed = data.where('done'), data.where('pending'), data.where('failed')
    U = problem.to_unit(data.X)
    scale = math.sqrt(problem.dim)
    loss = problem.sign * data.y  # lower is better; NaN unless done
    used = int(done.sum() + failed.sum())
    total = problem.total_budget
    summary = {
        'problem': {'name': problem.name, 'objective': problem.objective, 'direction': problem.direction,
                    'dim': problem.dim, 'noise': problem.noise},
        'counts': {'done': int(done.sum()), 'pending': int(pending.sum()), 'failed': int(failed.sum())},
        'budget': {'total': total, 'used': used, 'pending': int(pending.sum()),
                   'remaining': total - used - int(pending.sum())},
        'incumbent': None, 'top': [], 'blocks': [], 'blocks_since_improvement': None,
        'replicates': None, 'near_duplicate_pairs': 0, 'explored': {},
        'failures': {'count': int(failed.sum()), 'rate': _number(failed.sum() / max(used, 1)),
                     'by_block': {}},
    }
    if problem.pool is not None:
        taken = set(data.candidate[done | pending].tolist())
        summary['pool'] = {'file': problem.pool.file, 'size': len(problem.pool),
                           'tested': len(set(data.candidate[done | failed].tolist())),
                           'pending': len(set(data.candidate[pending].tolist())),
                           'available': len(problem.pool) - len(taken)}
    if done.any():
        order = np.flatnonzero(done)[np.argsort(loss[done], kind='stable')]
        best = order[0]
        summary['incumbent'] = {**_row(problem, data, best), 'at_bounds': _at_bounds(problem, data.X[best])}
        summary['top'] = [_row(problem, data, i) for i in order[:TOP]]

    # Blocks in order, each judged against the incumbent it started from.
    incumbent, since = None, 0
    for block in sorted(set(data.blocks.tolist())):
        rows = data.blocks == block
        entry = {'block': block, 'optimizers': list(dict.fromkeys(data.optimizer[rows].tolist())),
                 'done': int((rows & done).sum()), 'pending': int((rows & pending).sum()),
                 'failed': int((rows & failed).sum()), 'best': None, 'best_so_far': None,
                 'improved_by': None, 'spread': None, 'distance_to_previous_incumbent': None}
        if rows.sum() > 1:
            pairs = _distances(U[rows], U[rows])[np.triu_indices(int(rows.sum()), 1)]
            entry['spread'] = float(np.median(pairs) / scale)
        if incumbent is not None:
            entry['distance_to_previous_incumbent'] = float(
                np.median(_distances(U[rows], U[[incumbent]])) / scale)
        if (rows & done).any():
            candidates = np.flatnonzero(rows & done)
            local = candidates[np.argmin(loss[candidates])]
            entry['best'] = _number(data.y[local])
            if incumbent is None or loss[local] < loss[incumbent]:
                if incumbent is not None:
                    entry['improved_by'] = float(loss[incumbent] - loss[local])
                incumbent, since = local, 0
            else:
                entry['improved_by'] = 0.
                since += 1
            entry['best_so_far'] = _number(data.y[incumbent])
        summary['blocks'].append(entry)
        if entry['failed']:
            summary['failures']['by_block'][str(block)] = entry['failed']
    if incumbent is not None:
        summary['blocks_since_improvement'] = since

    # Replicates: done rows at exactly the same point.
    groups = {}
    for i in np.flatnonzero(done):
        groups.setdefault(tuple(data.X[i]), []).append(data.y[i])
    repeated = [values for values in groups.values() if len(values) > 1]
    if repeated:
        squares = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in repeated)
        degrees = sum(len(v) - 1 for v in repeated)
        summary['replicates'] = {'groups': len(repeated), 'rows': sum(map(len, repeated)),
                                 'pooled_sd': float(math.sqrt(squares / degrees))}

    placed = done | pending
    if placed.sum() > 1:
        close = _distances(U[placed], U[placed]) / scale < TOUCH
        summary['near_duplicate_pairs'] = int(np.triu(close, 1).sum())

    ran = done | failed
    for i, p in enumerate(problem.parameters):
        if ran.any():
            summary['explored'][p.name] = {'low': float(data.X[ran, i].min()), 'high': float(data.X[ran, i].max()),
                                           'unit_span': float(U[ran, i].max() - U[ran, i].min())}
    return summary


def model_diagnostics(problem, data, seed=0):
    """How well a GP fits the done rows, and which parameters it finds the loss
    most sensitive to (short lengthscales). None with fewer than 3 done rows."""
    done = data.where('done')
    if done.sum() < 3:
        return None
    from ..batch.gp import fit_gp  # torch is slow to import; only load it when asked
    try:
        _, fit = fit_gp(problem.to_unit(data.X[done]), problem.sign * data.y[done], seed)
    except Exception as exc:  # a degenerate table should not stop a summary
        return {'error': f'{type(exc).__name__}: {exc}'}
    ranking = sorted(zip(problem.names, fit['lengthscales']), key=lambda pair: pair[1])
    return {'n': fit['n'], 'loo_r2': fit['loo_r2'], 'noise_share': fit['noise_share'],
            'lengthscales': dict(ranking)}


def _g(value):
    return '-' if value is None else f'{value:.4g}'


def _point(row):
    point = ', '.join(f'{name}={value:.4g}' for name, value in row['parameters'].items())
    return f"candidate {row['candidate_id']} ({point})" if 'candidate_id' in row else point


def format_summary(summary):
    s, counts, budget = summary['problem'], summary['counts'], summary['budget']
    lines = [f"{s['name']}: {s['direction']} {s['objective']} over {s['dim']} parameters; noise {s['noise']}",
             f"experiments: {counts['done']} done, {counts['pending']} pending, {counts['failed']} failed"
             + f"; budget {budget['total']}: {budget['used']} used, {budget['pending']} pending, "
             f"{budget['remaining']} remaining"]
    if 'pool' in summary:
        pool = summary['pool']
        lines.append(f"pool: {pool['size']} candidates in {pool['file']}; {pool['tested']} tested, "
                     f"{pool['pending']} pending, {pool['available']} available")
    incumbent = summary['incumbent']
    if incumbent is None:
        lines.append('incumbent: none yet (no done experiments)')
    else:
        lines.append(f"incumbent: id {incumbent['id']} (block {incumbent['block']}), "
                     f"{s['objective']} {_g(incumbent['value'])} at {_point(incumbent)}")
        if incumbent['at_bounds']:
            lines.append('  at bounds: ' + ', '.join(f"{b['parameter']} {b['side']}" for b in incumbent['at_bounds']))
    if summary['blocks']:
        header = ['block', 'optimizers', 'done', 'pend', 'fail', 'best', 'best so far', 'improved by',
                  'spread', 'to prev. incumbent']
        rows = [[str(b['block']), '+'.join(b['optimizers']), str(b['done']), str(b['pending']), str(b['failed']),
                 _g(b['best']), _g(b['best_so_far']), _g(b['improved_by']), _g(b['spread']),
                 _g(b['distance_to_previous_incumbent'])] for b in summary['blocks']]
        widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
        lines.append('blocks:')
        lines += ['  ' + '  '.join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in [header, *rows]]
    if summary['blocks_since_improvement'] is not None:
        lines.append(f"blocks since the incumbent improved: {summary['blocks_since_improvement']}")
    if summary['top']:
        lines.append(f'top {len(summary["top"])}:')
        lines += [f"  id {r['id']} (block {r['block']}): {_g(r['value'])} at {_point(r)}"
                  for r in summary['top']]
    if summary['replicates']:
        r = summary['replicates']
        lines.append(f"replicates: {r['groups']} groups ({r['rows']} rows), pooled sd {_g(r['pooled_sd'])}")
    if summary['near_duplicate_pairs']:
        lines.append(f"near-duplicate pairs (done or pending): {summary['near_duplicate_pairs']}")
    if summary['explored']:
        lines.append('explored (done or failed): ' + '; '.join(
            f"{name} {_g(e['low'])} to {_g(e['high'])} ({e['unit_span']:.0%} of range)"
            for name, e in summary['explored'].items()))
    model = summary.get('model')
    if model and 'error' in model:
        lines.append(f"model: GP fit failed ({model['error']})")
    elif model:
        lines.append(f"model (GP on {model['n']} done rows): leave-one-out R2 {model['loo_r2']:.2f}, "
                     f"noise {model['noise_share']:.0%} of variance; lengthscales, most sensitive first: "
                     + ', '.join(f'{name} {value:.2g}' for name, value in model['lengthscales'].items()))
    if summary['failures']['count']:
        f = summary['failures']
        lines.append(f"failures: {f['count']} ({f['rate']:.0%} of runs), by block "
                     + ', '.join(f'{b}: {n}' for b, n in f['by_block'].items()))
    return '\n'.join(lines)
