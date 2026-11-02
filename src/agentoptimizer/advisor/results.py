"""Results coming in: measured pending rows, manual experiments and prior data.

Both entry points are all-or-nothing: the updated table is checked as a whole
and written only if it is valid, so a bad results file changes nothing.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .data import check_table, load_campaign, write_table


class ResultsError(ValueError):
    """A results or import file cannot be applied; the message lists every problem."""


def _read(path):
    path = Path(path)
    if not path.exists():
        raise ResultsError(f'{path} not found')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = [c.strip() for c in reader.fieldnames or []]
        rows = [{k.strip(): (v or '').strip() for k, v in row.items() if k is not None} for row in reader]
    return columns, [row for row in rows if any(row.values())]


def _status(problem, row, line, errors):
    status = row.get('status') or ('done' if row.get(problem.objective) else '')
    if status not in ('done', 'failed'):
        errors.append(f'line {line}: give a {problem.objective} value, or status failed'
                      + (f' (status {status!r} is not allowed here)' if status else ''))
    return status


def _new_row(problem, row, identifier, block, source, status):
    values = {name: row.get(name, '') for name in problem.names}
    extra = {}
    if problem.pool is not None:
        # Values a pool row leaves out are the candidate's own.
        extra['candidate_id'] = row.get('candidate_id', '')
        position = problem.pool.rows().get(extra['candidate_id'])
        if position is not None:
            for name, value in zip(problem.names, problem.pool.X[position]):
                values[name] = values[name] or repr(float(value))
    return {'id': str(identifier), 'block': str(block), 'status': status, 'source': source,
            'optimizer': row.get('optimizer') or source, **extra, **values,
            problem.objective: row.get(problem.objective, ''), 'notes': row.get('notes', '')}


def _pool_columns(problem):
    return ['candidate_id'] if problem.pool is not None else []


def _commit(campaign, problem, table, errors):
    if 'notes' not in table.columns:
        table.columns.append('notes')
    errors += [str(issue) for issue in check_table(problem, table) if issue.level == 'error']
    if errors:
        raise ResultsError('nothing was written:\n' + '\n'.join(errors))
    write_table(campaign, table)


def record_results(campaign, path):
    """Fill in pending rows by id; rows without an id are manual experiments.
    Returns (ids recorded, ids added)."""
    problem, table, data = load_campaign(campaign)
    columns, rows = _read(path)
    allowed = ['id', problem.objective, 'status', 'notes', 'optimizer', *_pool_columns(problem), *problem.names]
    unknown = [c for c in columns if c not in allowed]
    if unknown:
        raise ResultsError(f'{path}: unknown column(s) {", ".join(unknown)}; allowed: {", ".join(allowed)}')
    by_id = {row['id']: row for row in table.rows}
    block = int(data.blocks.max()) if len(data.blocks) else 0
    next_id = int(data.ids.max()) + 1 if len(data.ids) else 1
    errors, recorded, added, seen = [], [], [], set()
    for line, row in enumerate(rows, start=2):
        status = _status(problem, row, line, errors)
        identifier = row.get('id')
        if not identifier:
            table.rows.append(_new_row(problem, row, next_id, block, 'manual', status))
            added.append(next_id)
            next_id += 1
            continue
        target = by_id.get(identifier)
        if identifier in seen:
            errors.append(f'line {line}: id {identifier} appears twice in {path}')
        elif target is None:
            errors.append(f'line {line}: data.csv has no id {identifier}')
        elif target['status'] != 'pending':
            errors.append(f'line {line}: id {identifier} is {target["status"]}, not pending; '
                          'change a recorded result by editing data.csv')
        else:
            for name in problem.names:
                if row.get(name) and _differs(row[name], target[name]):
                    errors.append(f'line {line}: id {identifier} has {name} {row[name]} here '
                                  f'but {target[name]} in data.csv')
            target['status'] = status
            target[problem.objective] = row.get(problem.objective, '')
            if row.get('notes'):
                target['notes'] = row['notes']
            recorded.append(int(identifier))
        seen.add(identifier)
    _commit(campaign, problem, table, errors)
    return recorded, added


def _differs(a, b):
    try:
        return float(a) != float(b)
    except ValueError:
        return a != b


def import_rows(campaign, path):
    """Add prior results from another source as block 0. Returns the new ids."""
    problem, table, data = load_campaign(campaign)
    columns, rows = _read(path)
    if 'id' in columns:
        raise ResultsError(f'{path}: import assigns ids; rename the id column (to notes, say) or drop it')
    allowed = [*_pool_columns(problem), *problem.names, problem.objective, 'status', 'notes', 'optimizer']
    unknown = [c for c in columns if c not in allowed]
    # From a pool, the candidate_id is enough: the values come from the pool.
    needed = ['candidate_id'] if problem.pool is not None else problem.names
    missing = [c for c in [*needed, problem.objective] if c not in columns]
    if unknown or missing:
        raise ResultsError(f'{path}: ' + '; '.join(part for part in [
            'missing column(s) ' + ', '.join(missing) if missing else '',
            'unknown column(s) ' + ', '.join(unknown) if unknown else ''] if part)
            + f'; allowed: {", ".join(allowed)}')
    if not rows:
        raise ResultsError(f'{path} has no rows')
    first = int(data.ids.max()) + 1 if len(data.ids) else 1
    errors = []
    for i, row in enumerate(rows):
        status = _status(problem, row, i + 2, errors)
        table.rows.append(_new_row(problem, row, first + i, 0, 'import', status))
    _commit(campaign, problem, table, errors)
    return list(range(first, first + len(rows)))
