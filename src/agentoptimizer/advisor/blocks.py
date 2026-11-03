"""Block folders: blocks/NNN/, one per decided block.

new_block() opens the next block and saves the summary its decision is based
on. The advisor writes propose.py there: the block's SIZE and a function
propose(campaign, q) that returns the next q points as a batch.Proposal. The
lab measures [budget] round_size points at a time, so a block runs in rounds:
run_block() calls propose() for the next round and writes proposals.csv and
round.json, validate_proposals() checks them, and accept_block() adds them to
data.csv as pending rows, once decisions.md records the block's decision.
Later rounds repeat run and accept on the data so far, with no new decision.
"""
from __future__ import annotations

import csv
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys

from .campaign import Campaign

from .data import Issue, check_candidate, check_point, check_table, format_number, load_campaign, write_table
from .summary import model_diagnostics, summarize

# Packages a proposal script may use, reported with each block's summary.
ENVIRONMENT = ('numpy', 'scipy', 'torch', 'botorch', 'gpytorch', 'cma', 'Py-BOBYQA', 'agentoptimizer')
# Columns a proposals.csv may have besides the parameters.
PROPOSAL_EXTRAS = ('optimizer', 'notes')
# Files run_block() writes into the block folder.
OUTPUTS = ('proposals.csv', 'round.json', 'propose.log')
# propose.py imports the agentoptimizer that runs this code, also when that is
# a copy outside any environment (replay puts one in the sandbox).
LIBRARY = str(Path(__file__).resolve().parents[2])
BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv.pop(1)); "
             "from agentoptimizer.advisor.blocks import propose_round; propose_round()")


class BlockError(ValueError):
    """A block cannot be opened, run or accepted; the message says why."""


def block_folder(campaign, number):
    return Path(campaign) / 'blocks' / f'{number:03d}'


def block_size(campaign, number):
    """The block's SIZE, as its last run recorded it; None before the first run."""
    planned = read_round(campaign, number)
    return planned['size'] if planned else None


def current_block(campaign, data=None):
    """The block taking proposals: the last one in data.csv while it has rounds
    left, else the next one."""
    data = data if data is not None else load_campaign(campaign)[2]
    last = int(data.blocks.max()) if len(data.blocks) else 0
    size = block_size(campaign, last) if last else None
    return last if size and data.in_block(last) < size else last + 1


def new_block(campaign):
    """Open the next block (or reopen it, if it was never accepted) and save its summary."""
    problem, _, data = load_campaign(campaign)
    number = current_block(campaign, data)
    if data.in_block(number):
        raise BlockError(f'block {number} still has rounds to run ({data.in_block(number)} of '
                         f'{block_size(campaign, number)} experiments so far); continue it with run and accept')
    folder = block_folder(campaign, number)
    folder.mkdir(parents=True, exist_ok=True)
    summary = {'block': number, **summarize(problem, data),
               'model': model_diagnostics(problem, data), 'environment': environment()}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return number, folder


def environment():
    """Versions a proposal script runs with, so a decision can name its defaults."""
    versions = {'python': platform.python_version()}
    for name in ENVIRONMENT:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def _number(campaign, number):
    return number if number is not None else current_block(campaign)


def round_points(problem, data, number, size):
    """How many points the block's next round has: the round size, cut to what
    is left of the block and the budget. A first round needs the whole block
    within the budget."""
    done = data.in_block(number)
    remaining = problem.total_budget - len(data.ids)
    if done == 0 and size > remaining:
        raise BlockError(f'SIZE = {size} exceeds the remaining budget of {remaining}')
    if done >= size or remaining < 1:
        raise BlockError(f'block {number} is complete ({done} of {size}, {remaining} left in the budget); '
                         'open the next with new-block')
    return min(problem.round_size, size - done, remaining)


def propose_round():
    """Run from the block folder, in a fresh interpreter: propose the next round
    with propose.py and write proposals.csv and round.json."""
    from ..batch import Proposal  # imports torch: only where a script runs
    script = runpy.run_path('propose.py', run_name='propose')
    size, propose = script.get('SIZE'), script.get('propose')
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise BlockError(f'propose.py must set SIZE, the block size, to a positive integer, not {size!r}')
    if not callable(propose):
        raise BlockError('propose.py must define propose(campaign, q)')
    campaign = Campaign.load('propose.py')
    q = round_points(campaign.problem, campaign.data, campaign.block, size)
    proposal = propose(campaign, q)
    if not isinstance(proposal, Proposal) or len(proposal.X) != q:
        raise BlockError(f'propose(campaign, {q}) must return a batch.Proposal of {q} points, not {proposal!r}')
    campaign.write(proposal)
    Path('round.json').write_text(json.dumps({'size': size, 'round': campaign.round, 'q': q}) + '\n')


def _execute(folder, timeout):
    # Stale files must not pass for this run's output.
    for name in ['proposals.csv', 'round.json']:
        (folder / name).unlink(missing_ok=True)
    try:
        result = subprocess.run([sys.executable, '-I', '-c', BOOTSTRAP, LIBRARY], cwd=folder,
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise BlockError(f'propose.py ran longer than {timeout:g} s') from None
    (folder / 'propose.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        tail = '\n'.join((result.stdout + result.stderr).strip().splitlines()[-15:])
        raise BlockError(f'propose.py exited with {result.returncode}; see {folder / "propose.log"}\n{tail}')
    path = folder / 'proposals.csv'
    return path.read_text() if path.exists() else None


def run_block(campaign, number=None, timeout=600, verify=True):
    """Propose the block's next round with this interpreter, then validate it.
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
    required = [*(['candidate_id'] if pool is not None else problem.names), 'optimizer']
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
        if not row['optimizer']:
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
    remaining = problem.total_budget - int(
        (data.where('done') | data.where('failed') | data.where('pending')).sum())
    if len(rows) > remaining:
        issues.append(Issue('error', 'proposals.csv', f'{len(rows)} proposals exceed the remaining budget of {remaining}'))
    planned = read_round(campaign, number)
    if planned is None:
        issues.append(Issue('error', 'round.json', 'not found; run propose.py with opt_run.py run'))
    elif len(rows) != planned['q']:
        issues.append(Issue('error', 'proposals.csv', f'has {len(rows)} rows; this round has {planned["q"]}'))
    elif data.in_block(number) + len(rows) > planned['size']:
        issues.append(Issue('error', 'proposals.csv', f'would take block {number} past its SIZE of {planned["size"]}'))
    return rows, sorted(issues, key=lambda issue: issue.level != 'error')


def read_round(campaign, number):
    """What the block's last run proposed: {'size', 'round', 'q'}, or None."""
    path = block_folder(campaign, number) / 'round.json'
    return json.loads(path.read_text()) if path.exists() else None


def has_decision(campaign, number):
    path = Path(campaign) / 'decisions.md'
    return path.exists() and re.search(rf'^##\s+Block\s+0*{number}\b', path.read_text(), re.M) is not None


def accept_block(campaign, number=None):
    """Append the round's proposals to data.csv as pending rows; returns their ids."""
    problem, table, data = load_campaign(campaign)
    number = _number(campaign, number)
    if number != current_block(campaign, data):
        raise BlockError(f'block {number} is not the block taking proposals ({current_block(campaign, data)})')
    if not has_decision(campaign, number):
        raise BlockError(f'decisions.md has no "## Block {number}" entry; '
                         'record the decision before accepting its proposals')
    rows, issues = validate_proposals(campaign, number)
    errors = [issue for issue in issues if issue.level == 'error']
    if errors:
        raise BlockError('proposals are not valid:\n' + '\n'.join(map(str, errors)))
    first = int(data.ids.max()) + 1 if len(data.ids) else 1
    for i, row in enumerate(rows):
        table.rows.append({
            'id': str(first + i), 'block': str(number), 'status': 'pending', 'source': 'advisor',
            'optimizer': row['optimizer'],
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
