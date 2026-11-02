"""Launch agent campaigns in the background and report on them.

    launch  start one detached run_agent.py per seed; return once each has
            passed the isolation preflight, printing its run directory
    wait    block until the given runs end, then report on them
    status  one line per run under runs/, or a full report on the given runs
    tasks   list the task keys

Runs started by `launch` keep going when this script, the terminal or a
Claude Code session ends. Each run's console output is in
private/launcher.log. Reading a run never touches its workspace: the
evaluator's journal is opened read-only.

Usage
-----
    .venv/bin/python scripts/agent_runs.py launch --task sphere10 --budget 100 --seeds 0 1 2
    .venv/bin/python scripts/agent_runs.py launch --task mof_pool --budget 100 --agent claude
    .venv/bin/python scripts/agent_runs.py status
    .venv/bin/python scripts/agent_runs.py wait runs/run-1a2b3c4d5e6f
    .venv/bin/python scripts/agent_runs.py status runs/run-1a2b3c4d5e6f --baselines runs/baselines

Flags that `launch` does not know (--agent, --model, --effort, ...) go to
run_agent.py unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys
import time

from agentoptimizer.blind_runner.evaluator import incumbents

REPO = Path(__file__).resolve().parents[1]
# A run with no launcher record counts as live while its journal or event
# log changed this recently.
ACTIVE_SECONDS = 600


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------- launch

def wait_for_preflight(process, log, timeout):
    """Follow a starting run's console output until its preflight passes.
    Returns (run directory or None, output so far, passed)."""
    root, output = None, ''
    deadline = time.time() + timeout
    with log.open() as stream:
        while time.time() < deadline:
            output += stream.read()
            match = re.search(r'^Run directory: (.+)$', output, re.M)
            if match and root is None:
                root = Path(match.group(1))
            if 'Isolation passed' in output:
                return root, output, True
            if process.poll() is not None:
                output += stream.read()
                return root, output, False
            time.sleep(.5)
    return root, output, False


def launch(args, passthrough):
    runs_dir = Path(args.runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    started = []
    for seed in args.seeds:
        command = [sys.executable, str(REPO / 'scripts/run_agent.py'), '--task', args.task,
                   '--budget', str(args.budget), '--seed', str(seed),
                   '--runs-dir', str(runs_dir), *passthrough]
        # The run directory's name is only known once the runner prints it.
        log = runs_dir / f'.launch-{os.getpid()}-{seed}.log'
        with log.open('w') as out:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=out,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        started.append((seed, command, process, log))
    failures = 0
    for seed, command, process, log in started:
        root, output, passed = wait_for_preflight(process, log, args.timeout)
        if root is not None and (root / 'private').is_dir():
            log.rename(root / 'private/launcher.log')
            (root / 'private/launcher.json').write_text(json.dumps(
                {'pid': process.pid, 'command': command, 'started': time.time()}, indent=2) + '\n')
        if passed:
            print(f'seed {seed}: {relative(root)} (preflight passed, pid {process.pid})')
            continue
        failures += 1
        state = 'still starting' if process.poll() is None else f'exited {process.returncode}'
        where = relative(root / 'private/launcher.log') if root else relative(log)
        print(f'seed {seed}: preflight did not pass ({state}); log: {where}')
        print('\n'.join('    ' + line for line in output.strip().splitlines()[-15:]))
    return 1 if failures else 0


# ---------------------------------------------------------------- reading runs

def state(root):
    private = root / 'private'
    audit = read_json(private / 'integrity_audit.json')
    if audit is not None:
        if (audit.get('unknown_evaluations') or not audit.get('artifacts_match_journal')
                or not audit.get('only_workspace_execution_tools')):
            return 'audit failed'
        exit_record = read_json(private / 'cli_exit.json')
        if exit_record is None:
            return 'check only'
        code = exit_record['returncode']
        return 'finished' if code == 0 else f'agent exited {code}'
    launcher = read_json(private / 'launcher.json')
    if launcher is not None:
        return 'running' if alive(launcher['pid']) else 'stopped'
    return 'running?' if time.time() - last_activity(private) < ACTIVE_SECONDS else 'stopped'


def last_activity(private):
    times = [p.stat().st_mtime for p in private.glob('*') if p.is_file()]
    return max(times, default=0.)


def progress(root):
    """Evaluations used, the incumbent's value in the scoring convention
    (noise-free, feasible), and the method blocks as [label, evaluations]."""
    path = root / 'private/evaluations.sqlite'
    if not path.exists():
        return 0, None, []
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        used = db.execute("SELECT COUNT(*) FROM evaluations WHERE campaign='run'").fetchone()[0]
        best, blocks = None, []
        for row, _, observed, _, true_value in incumbents(db, 'run'):
            best = true_value if true_value is not None else observed
            label = row[5]
            if blocks and blocks[-1][0] == label:
                blocks[-1][1] += 1
            else:
                blocks.append([label, 1])
    finally:
        db.close()
    return used, best, blocks


def summary(root):
    private = root / 'private'
    manifest = read_json(private / 'manifest.json') or {}
    scoring = read_json(private / 'scoring.json') or {}
    used, best, blocks = progress(root)
    f_opt = scoring.get('f_opt')
    return {
        'run': root.name, 'root': root, 'task': manifest.get('task', '?'),
        'task_name': scoring.get('task_name'), 'seed': manifest.get('seed', '?'),
        'agent': manifest.get('agent', '?'), 'model': manifest.get('model', '?'),
        'effort': manifest.get('reasoning_effort', '?'),
        'budget': manifest.get('budget', 0), 'used': used, 'best': best,
        'gap': None if best is None or f_opt is None else best - f_opt,
        'state': state(root), 'blocks': blocks,
        'started': (private / 'manifest.json').stat().st_mtime if (private / 'manifest.json').exists() else 0.,
    }


def number(value):
    return '-' if value is None else f'{value:.4g}'


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def table(summaries):
    header = ['run', 'task', 'seed', 'agent', 'evals', 'state', 'best', 'gap', 'started']
    rows = [[s['run'], s['task'], str(s['seed']), f"{s['agent']}/{s['model']}/{s['effort']}",
             f"{s['used']}/{s['budget']}", s['state'], number(s['best']), number(s['gap']),
             time.strftime('%Y-%m-%d %H:%M', time.localtime(s['started']))] for s in summaries]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print('  '.join(cell.ljust(width) for cell, width in zip(r, widths)).rstrip())


def baseline_finals(outdir, task_name, budget):
    """Median over seeds of each baseline optimizer's best-so-far at `budget`."""
    finals = {}
    for folder in sorted((Path(outdir) / task_name).glob('*')):
        values = []
        for trace in folder.glob('seed_*/trace.csv'):
            with trace.open() as stream:
                rows = [r for r in csv.DictReader(stream) if int(r['eval_idx']) <= budget]
            if rows and rows[-1]['y_best_so_far'] != '':
                values.append(float(rows[-1]['y_best_so_far']))
        if values:
            finals[folder.name] = (statistics.median(values), len(values))
    return finals


def report(roots, baselines=None):
    summaries = [summary(root) for root in roots]
    table(summaries)
    for s in summaries:
        private = s['root'] / 'private'
        print(f"\n{s['run']}  ({relative(s['root'])})")
        print(f"  task {s['task']}, seed {s['seed']}, {s['agent']} {s['model']} effort {s['effort']}")
        print(f"  {s['state']}: {s['used']}/{s['budget']} evaluations, best {number(s['best'])}, "
              f"gap to optimum {number(s['gap'])}")
        if s['blocks']:
            print('  blocks: ' + ' -> '.join(f'{label} x{n}' for label, n in s['blocks']))
        audit = read_json(private / 'integrity_audit.json')
        if audit is not None:
            print(f"  audit: journal matches artifacts {audit['artifacts_match_journal']}, "
                  f"unknown evaluations {audit['unknown_evaluations']}, "
                  f"only sandbox tools {audit['only_workspace_execution_tools']} ({audit['tool_calls']} calls)")
        if baselines and s['task_name'] and s['best'] is not None:
            finals = baseline_finals(baselines, s['task_name'], s['budget'])
            if finals:
                beaten = sum(value > s['best'] for value, _ in finals.values())
                print(f"  baselines at {s['budget']} evaluations (median over seeds): "
                      f"agent beats {beaten} of {len(finals)}")
                for name, (value, n) in sorted(finals.items(), key=lambda item: item[1][0]):
                    print(f'    {name:<14} {number(value)}  ({n} seeds)')
            else:
                print(f"  baselines: none for this task under {baselines}")
        final = (private / 'agent_final.txt')
        if final.exists() and final.read_text().strip():
            lines = final.read_text().strip().splitlines()
            print("  agent's final message:")
            print('\n'.join('    ' + line for line in lines[:20]))
            if len(lines) > 20:
                print(f'    ... {len(lines) - 20} more lines in {relative(final)}')
        if s['state'] not in ('finished', 'running', 'running?', 'check only'):
            print(f"  logs: {relative(private)}/launcher.log, cli_stderr.log, evaluator_errors.log")
    return summaries


def run_dirs(paths, runs_dir):
    if paths:
        roots = [Path(p).resolve() for p in paths]
        missing = [str(p) for p in roots if not (p / 'private').is_dir()]
        if missing:
            sys.exit('Not a run directory: ' + ', '.join(missing))
        return roots
    roots = [p for p in Path(runs_dir).glob('run-*') if (p / 'private/manifest.json').exists()]
    return sorted(roots, key=lambda p: (p / 'private/manifest.json').stat().st_mtime, reverse=True)


# ---------------------------------------------------------------- commands

def status(args):
    roots = run_dirs(args.runs, args.runs_dir)
    if not roots:
        print(f'No runs under {relative(args.runs_dir)}.')
        return 0
    if args.runs:
        report(roots, args.baselines)
    else:
        table([summary(root) for root in roots[:None if args.all else args.limit]])
        if not args.all and len(roots) > args.limit:
            print(f'... {len(roots) - args.limit} older runs; use --all')
    return 0


def wait(args):
    roots = run_dirs(args.runs, args.runs_dir)
    while any(state(root) in ('running', 'running?') for root in roots):
        time.sleep(args.interval)
    summaries = report(roots, args.baselines)
    return 0 if all(s['state'] in ('finished', 'check only') for s in summaries) else 1


def tasks(_):
    from agentoptimizer.tasks.registry import TASKS
    print('\n'.join(sorted(TASKS)))
    print('\nDescriptions: docs/TASKS.md')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir', type=Path, default=REPO / 'runs')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('launch', help='start one detached run per seed')
    p.add_argument('--task', required=True)
    p.add_argument('--budget', required=True, type=int)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--timeout', type=float, default=600, help='seconds to wait for each preflight')
    p = commands.add_parser('status', help='list runs, or report on the given ones')
    p.add_argument('runs', nargs='*')
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--all', action='store_true')
    p.add_argument('--baselines', type=Path, help='run_sweep.py output directory to compare against')
    p = commands.add_parser('wait', help='block until the given runs end, then report')
    p.add_argument('runs', nargs='+')
    p.add_argument('--interval', type=float, default=30)
    p.add_argument('--baselines', type=Path)
    commands.add_parser('tasks', help='list task keys')
    args, passthrough = parser.parse_known_args(argv)
    if args.command == 'launch':
        return launch(args, passthrough)
    if passthrough:
        parser.error('unrecognized arguments: ' + ' '.join(passthrough))
    return {'status': status, 'wait': wait, 'tasks': tasks}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
