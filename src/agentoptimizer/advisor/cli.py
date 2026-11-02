"""Command line for advisor mode: scripts/opt_run.py <command> <campaign folder>.

    init       write a problem.toml template to fill in
    check      validate problem.toml and data.csv
    summarize  print the data summary the advisor reads (--json for the raw form)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import DataError, check_table, parse_table, read_table
from .problem import TEMPLATE, ProblemError, load_problem
from .summary import format_summary, summarize


def load(folder):
    """The campaign's problem, table and parsed data; raises on any error."""
    problem = load_problem(folder)
    table = read_table(folder, problem)
    return problem, table, parse_table(problem, table)


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
    budget = ', '.join(part for part in [
        f'{problem.total_budget} experiments' if problem.total_budget else '',
        f'blocks of {problem.block_size}' if problem.block_size else ''] if part) or 'not set'
    lines = [f'{problem.name}: {problem.direction} {problem.objective}; noise {problem.noise}; budget {budget}']
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
    problem, _, data = load(args.campaign)
    result = summarize(problem, data)
    print(json.dumps(result, indent=2) if args.json else format_summary(result))
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
    command('summarize', summary, 'print the data summary').add_argument(
        '--json', action='store_true', help='print the summary as JSON')
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ProblemError, DataError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
