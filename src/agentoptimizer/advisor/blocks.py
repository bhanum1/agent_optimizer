"""Block folders: blocks/NNN/, one per proposed block.

new_block() opens the next block and saves the summary its decision is based
on. The advisor writes propose.py there; run_block() runs it and
validate_proposals() checks the proposals.csv it wrote. accept_block() adds
the proposals to data.csv as pending rows, once decisions.md records the
decision for that block.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
import sys

from .data import Issue, check_candidate, check_point, check_table, format_number, load_campaign, write_table
from .summary import summarize

# Columns a proposals.csv may have besides the parameters.
PROPOSAL_EXTRAS = ('optimizer', 'notes')


class BlockError(ValueError):
    """A block cannot be opened, run or accepted; the message says why."""


def block_folder(campaign, number):
    return Path(campaign) / 'blocks' / f'{number:03d}'


def open_block_number(data):
    """The block after the last one in data.csv: the block being prepared."""
    return int(data.blocks.max()) + 1 if len(data.blocks) else 1


def new_block(campaign):
    """Open the next block (or reopen it, if it was never accepted) and save its summary."""
    problem, _, data = load_campaign(campaign)
    number = open_block_number(data)
    folder = block_folder(campaign, number)
    folder.mkdir(parents=True, exist_ok=True)
    summary = {'block': number, **summarize(problem, data)}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return number, folder


def _number(campaign, number):
    if number is not None:
        return number
    return open_block_number(load_campaign(campaign)[2])


def _execute(folder, timeout):
    # A stale file must not pass for this run's output.
    (folder / 'proposals.csv').unlink(missing_ok=True)
    try:
        result = subprocess.run([sys.executable, '-I', 'propose.py'], cwd=folder, capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise BlockError(f'propose.py ran longer than {timeout:g} s') from None
    (folder / 'propose.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        tail = '\n'.join((result.stdout + result.stderr).strip().splitlines()[-15:])
        raise BlockError(f'propose.py exited with {result.returncode}; see {folder / "propose.log"}\n{tail}')
    path = folder / 'proposals.csv'
    return path.read_text() if path.exists() else None


def run_block(campaign, number=None, timeout=600, verify=True):
    """Run the block's propose.py with this interpreter, then validate its proposals.
    With verify, run it twice: a script whose proposals differ between runs
    cannot be reproduced later, so it is an error."""
    number = _number(campaign, number)
    folder = block_folder(campaign, number)
    if not (folder / 'propose.py').exists():
        raise BlockError(f'{folder / "propose.py"} not found; write it first')
    first = _execute(folder, timeout)
    if verify and first is not None and _execute(folder, timeout) != first:
        raise BlockError('propose.py wrote different proposals on a second run; '
                         'seed every random number generator (numpy, torch, the optimizer)')
    return validate_proposals(campaign, number)


def validate_proposals(campaign, number=None):
    """The block's proposals (rows of text) and every problem with them."""
    problem, _, data = load_campaign(campaign)
    number = _number(campaign, number)
    path = block_folder(campaign, number) / 'proposals.csv'
    if not path.exists():
        return [], [Issue('error', str(path), 'not found; run propose.py first')]
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = [{k: (v or '').strip() for k, v in row.items() if k is not None} for row in reader]
    issues = []
    pool = problem.pool
    if pool is not None:
        # A pool proposal names a candidate; its values come from the pool.
        positions = pool.rows()
        for row in rows:
            if row.get('candidate_id') in positions:
                for name, value in zip(problem.names, pool.X[positions[row['candidate_id']]]):
                    row.setdefault(name, repr(float(value)))
    required = ['candidate_id'] if pool is not None else problem.names
    extras = [*PROPOSAL_EXTRAS, *(problem.names if pool is not None else []), 'candidate_id']
    missing = [name for name in required if name not in columns]
    unknown = [c for c in columns if c not in problem.names and c not in extras]
    if missing or unknown:
        message = '; '.join(part for part in [
            'missing column(s) ' + ', '.join(missing) if missing else '',
            'unknown column(s) ' + ', '.join(unknown) if unknown else ''] if part)
        return rows, [Issue('error', 'proposals.csv header', message)]
    if not rows:
        return rows, [Issue('error', 'proposals.csv', 'has no rows')]

    def identity(point, candidate):
        return candidate if pool is not None else tuple(point)

    known = {}
    for i in range(len(data.ids)):
        if data.status[i] in ('done', 'pending'):
            known.setdefault(identity(data.X[i], data.candidate[i]), (int(data.ids[i]), data.status[i]))
    seen = {}
    for line, row in enumerate(rows, start=2):
        where = f'proposals.csv line {line}'
        point, problems = check_point(problem, row)
        problems += check_candidate(problem, row, point)
        issues += [Issue('error', f'{where}, {column}', message) for column, message in problems]
        if 'optimizer' in columns and not row['optimizer']:
            issues.append(Issue('error', f'{where}, optimizer', 'is empty'))
        if problems:
            continue
        key = identity(point, row.get('candidate_id'))
        if key in seen:
            issues.append(Issue('error', where, f'repeats line {seen[key]} of proposals.csv'))
        elif key in known and known[key][1] == 'pending':
            issues.append(Issue('error', where, f'repeats id {known[key][0]}, which is still pending'))
        elif key in known:
            issues.append(Issue('warning', where, f'repeats id {known[key][0]} (a replicate); fine if intended'))
        seen.setdefault(key, line)
    remaining = problem.total_budget and problem.total_budget - int(
        (data.where('done') | data.where('failed') | data.where('pending')).sum())
    if problem.total_budget and len(rows) > remaining:
        issues.append(Issue('error', 'proposals.csv', f'{len(rows)} proposals exceed the remaining budget of {remaining}'))
    return rows, sorted(issues, key=lambda issue: issue.level != 'error')


def has_decision(campaign, number):
    path = Path(campaign) / 'decisions.md'
    return path.exists() and re.search(rf'^##\s+Block\s+0*{number}\b', path.read_text(), re.M) is not None


def accept_block(campaign, number=None, optimizer=None):
    """Append the block's proposals to data.csv as pending rows; returns their ids."""
    problem, table, data = load_campaign(campaign)
    number = _number(campaign, number)
    if (data.blocks == number).any():
        raise BlockError(f'block {number} is already in data.csv')
    if not has_decision(campaign, number):
        raise BlockError(f'decisions.md has no "## Block {number}" entry; '
                         'record the decision before accepting its proposals')
    rows, issues = validate_proposals(campaign, number)
    errors = [issue for issue in issues if issue.level == 'error']
    if errors:
        raise BlockError('proposals are not valid:\n' + '\n'.join(map(str, errors)))
    if not optimizer and not all(row.get('optimizer') for row in rows):
        raise BlockError('proposals.csv has no optimizer column; pass the method label')
    first = int(data.ids.max()) + 1 if len(data.ids) else 1
    for i, row in enumerate(rows):
        table.rows.append({
            'id': str(first + i), 'block': str(number), 'status': 'pending', 'source': 'advisor',
            'optimizer': row.get('optimizer') or optimizer,
            **({'candidate_id': row['candidate_id']} if problem.pool is not None else {}),
            **{p.name: format_number(p, float(row[p.name])) for p in problem.parameters},
            problem.objective: '', 'notes': row.get('notes', '')})
    if 'notes' not in table.columns:
        table.columns.append('notes')
    problems = [issue for issue in check_table(problem, table) if issue.level == 'error']
    if problems:
        raise BlockError('accepting would make data.csv invalid:\n' + '\n'.join(map(str, problems)))
    write_table(campaign, table)
    return list(range(first, first + len(rows)))
