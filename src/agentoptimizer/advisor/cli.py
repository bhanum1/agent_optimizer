"""Command line for advisor mode: scripts/opt_run.py <command> <campaign folder>.

    init       write a problem.toml template to fill in
    check      validate problem.toml and data.csv
    summarize  print the data summary the advisor reads (--json for the raw form)
    new-block  open the next block folder and save its summary
    run        propose the block's next round with propose.py, twice (it must be
               deterministic), then validate
    validate   validate the round's proposals.csv
    accept     add the round's proposals to data.csv as pending rows
    record     fill in pending rows from a results CSV (id, value or status failed);
               rows without an id are added as manual experiments
    import     add prior results from a CSV as block 0

A block's flow: new-block -> write "## Block N" in decisions.md and
blocks/NNN/propose.py -> run -> accept -> run the experiments -> record.
A block of SIZE points runs in rounds of [budget] round_size: after each
record, run and accept propose the next round, until the block is full.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .blocks import (BlockError, accept_block, current_block, new_block, read_round, run_block,
                     validate_proposals)
from .data import DataError, check_table, load_campaign, read_table
from .problem import TEMPLATE, ProblemError, load_problem
from .results import ResultsError, import_rows, record_results
from .summary import format_summary, model_diagnostics, summarize


def init(args):
    folder = Path(args.campaign)
    path = folder / 'problem.toml'
    if path.exists():
        sys.exit(f'{path} already exists')
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE)
    print(f'Wrote {path}. Fill it in, then run: opt_run.py check {folder}')
    return 0


def describe(problem):
    budget = (f'{problem.total_budget} experiments, blocks of {problem.block_size}, '
              f'rounds of {problem.round_size}')
    lines = [f'{problem.name}: {problem.direction} {problem.objective}; noise {problem.noise}; budget {budget}']
    if problem.pool is not None:
        lines.append(f'pool: {len(problem.pool)} candidates in {problem.pool.file}')
    for p in problem.parameters:
        scale = 'log' if p.log else 'linear'
        unit = f' {p.unit}' if p.unit else ''
        lines.append(f'  {p.name:<20} {p.type:<10} [{p.low:g}, {p.high:g}]{unit} ({scale})')
    return '\n'.join(lines)


def check(args):
    problem = load_problem(args.campaign)
    print(describe(problem))
    table = read_table(args.campaign, problem)
    if not table.rows:
        print('data.csv: no experiments yet')
        return 0
    counts = {s: sum(row['status'] == s for row in table.rows) for s in ('done', 'pending', 'failed')}
    print(f'data.csv: {len(table.rows)} rows (' + ', '.join(f'{n} {s}' for s, n in counts.items()) + ')')
    issues = check_table(problem, table)
    for issue in issues:
        print(issue)
    return 1 if any(issue.level == 'error' for issue in issues) else 0


def summary(args):
    problem, _, data = load_campaign(args.campaign)
    result = summarize(problem, data)
    if args.model:
        result['model'] = model_diagnostics(problem, data)
    print(json.dumps(result, indent=2) if args.json else format_summary(result))
    return 0


def new_block_command(args):
    number, folder = new_block(args.campaign)
    print(f'Block {number} is open: {folder} (summary.json saved).\n'
          f'Next: add "## Block {number}" to decisions.md, write {folder / "propose.py"}, '
          f'then run: opt_run.py run {args.campaign}')
    return 0


def show_proposals(rows, issues):
    if rows:
        columns = list(rows[0])
        widths = [max(len(c), *(len(r.get(c, '')) for r in rows)) for c in columns]
        print(f'{len(rows)} proposals:')
        for r in [dict(zip(columns, columns)), *rows]:
            print('  ' + '  '.join(r.get(c, '').ljust(w) for c, w in zip(columns, widths)).rstrip())
    for issue in issues:
        print(issue)
    return 1 if any(issue.level == 'error' for issue in issues) else 0


def run_command(args):
    number = args.block or current_block(args.campaign)
    rows, issues = run_block(args.campaign, number, args.timeout, verify=not args.no_verify)
    planned = read_round(args.campaign, number)
    if planned:
        print(f"Block {number}, round {planned['round']}: {planned['q']} of SIZE {planned['size']}.")
    return show_proposals(rows, issues)


def validate_command(args):
    return show_proposals(*validate_proposals(args.campaign, args.block))


def accept_command(args):
    number = args.block or current_block(args.campaign)
    ids = accept_block(args.campaign, number)
    planned, done = read_round(args.campaign, number), load_campaign(args.campaign)[2].in_block(number)
    left = (f'Block {number} is complete.' if done >= planned['size'] else
            f'Block {number} has {done} of {planned["size"]}; after recording, run and accept its next round.')
    print(f'Added ids {ids[0]}-{ids[-1]} to data.csv as pending. Run the experiments, then record the results. '
          + left)
    return 0


def record_command(args):
    recorded, added = record_results(args.campaign, args.file)
    print(f'Recorded {len(recorded)} pending row(s)' + (f' (ids {recorded[0]}-{recorded[-1]})' if recorded else '')
          + (f'; added {len(added)} manual row(s) (ids {added[0]}-{added[-1]})' if added else '') + '.')
    return 0


def import_command(args):
    ids = import_rows(args.campaign, args.file)
    print(f'Imported {len(ids)} row(s) as block 0 (ids {ids[0]}-{ids[-1]}).')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='opt_run.py', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    def command(name, function, text):
        sub = commands.add_parser(name, help=text)
        sub.add_argument('campaign', type=Path, help='campaign folder')
        sub.set_defaults(function=function)
        return sub

    command('init', init, 'write a problem.toml template')
    command('check', check, 'validate the campaign')
    sub = command('summarize', summary, 'print the data summary')
    sub.add_argument('--json', action='store_true', help='print the summary as JSON')
    sub.add_argument('--model', action='store_true', help='add GP diagnostics (lengthscales, leave-one-out fit)')
    command('new-block', new_block_command, 'open the next block')
    for name, function, text in [('run', run_command, "propose the block's next round and validate"),
                                 ('validate', validate_command, "validate the round's proposals"),
                                 ('accept', accept_command, "add the round's proposals as pending rows")]:
        sub = command(name, function, text)
        sub.add_argument('--block', type=int, help='block number (default: the open block)')
        if name == 'run':
            sub.add_argument('--timeout', type=float, default=600, help='seconds per run (default 600)')
            sub.add_argument('--no-verify', action='store_true',
                             help='run once, without checking that a second run gives the same proposals')
    command('record', record_command, 'fill in results for pending rows').add_argument(
        'file', type=Path, help='CSV with id and the objective (or status failed), optional notes')
    command('import', import_command, 'add prior results as block 0').add_argument(
        'file', type=Path, help='CSV with the parameters and the objective')
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ProblemError, DataError, BlockError, ResultsError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
