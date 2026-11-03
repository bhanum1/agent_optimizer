"""The workspace campaign: an anonymous advisor-mode folder for a registered task,
and the advisor code the sandbox runs it with.
"""
import csv
import io
from pathlib import Path
import shutil

from ..advisor.data import Table, standard_columns, write_table
from ..advisor.problem import load_problem

PACKAGE = Path(__file__).resolve().parents[1]

OPT_RUN = '''"""Advisor mode: manage the campaign folder. See --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from agentoptimizer.advisor.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
'''


def install_advisor(workspace):
    """Copy the advisor and batch code into workspace/lib, with an opt_run.py beside it.
    Both use only relative imports and the agent runtime's libraries, and name no task."""
    workspace = Path(workspace)
    library = workspace / 'lib/agentoptimizer'
    library.mkdir(parents=True)
    (library / '__init__.py').write_text('')
    for name in ['advisor', 'batch']:
        shutil.copytree(PACKAGE / name, library / name, ignore=shutil.ignore_patterns('__pycache__'))
    (workspace / 'opt_run.py').write_text(OPT_RUN)


def problem_toml(task, budget, block_size, round_size, seed):
    """An anonymous campaign for the task: x0..x{d-1}, minimize y."""
    indexed = getattr(task, 'indexed_pool', False)
    notes = ['An anonymous black-box objective. The names x0, x1, ... carry no meaning, and nothing '
             'is known about the function beyond the results in data.csv. Results come back after '
             'every round; the next decision comes when the block is complete.']
    if task.noisy:
        notes.append('Observations are noisy. The campaign is scored by the noise-free value at the '
                     'point with the lowest observed y.')
    if task.pool and not indexed:
        # A point maps to the nearest untested candidate, so a repeat differs.
        notes.append('The objective is not repeatable: the value returned for a point depends on which '
                     'points were already evaluated, so repeating a point generally returns a different value.')
    if indexed:
        notes.append('Experiments are chosen from candidates.csv only, and each candidate can be '
                     'evaluated once. Candidate ids are anonymous row numbers; constant features may exist.')
    notes.append(f'Campaign seed: {seed}. Derive the seeds of your scripts from it.')
    noise = 'unknown' if task.noisy or (task.pool and not indexed) else 'none'
    lines = ['name = "campaign"', 'description = """', *notes, '"""', f'noise = "{noise}"', '',
             '[objective]', 'name = "y"', 'direction = "minimize"', '',
             '[budget]', f'total = {budget}', f'block_size = {block_size}', f'round_size = {round_size}']
    if indexed:
        lines += ['', '[pool]', 'file = "candidates.csv"']
    for i, (low, high) in enumerate(task.bounds):
        lines += ['', '[[parameters]]', f'name = "x{i}"']
        if not indexed:
            # With a pool, the bounds are the candidates' range.
            lines += [f'low = {float(low)!r}', f'high = {float(high)!r}']
    return '\n'.join(lines) + '\n'


def candidates_csv(task):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['candidate_id', *(f'x{i}' for i in range(task.dim))])
    for i, row in enumerate(task.candidates):
        writer.writerow([i, *(repr(float(v)) for v in row)])
    return output.getvalue()


def build_campaign(root, task, budget, block_size, round_size, seed):
    """workspace/ and private/ for a run, with the empty campaign in both."""
    if task.n_constraints:
        raise ValueError('Constrained tasks are not supported yet.')
    workspace, private = Path(root) / 'workspace', Path(root) / 'private'
    for folder in [workspace / 'campaign', workspace / 'work', workspace / '.tmp', workspace / '.home',
                   private / 'campaign', private / 'blocks']:
        folder.mkdir(parents=True, exist_ok=True)
    install_advisor(workspace)
    truth = private / 'campaign'
    (truth / 'problem.toml').write_text(problem_toml(task, budget, block_size, round_size, seed))
    if getattr(task, 'indexed_pool', False):
        (truth / 'candidates.csv').write_text(candidates_csv(task))
    write_table(truth, Table(standard_columns(load_problem(truth))))
    (truth / 'decisions.md').write_text('')
    return workspace, private
