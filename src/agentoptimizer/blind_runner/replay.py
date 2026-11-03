"""Advisor replay: run advisor mode as a blind benchmark campaign.

The runner plays the lab. Each block, a fresh advisor call in the sandbox
proposes the next block in workspace/campaign/ exactly as /opt-run does; the
runner checks it, reproduces it, evaluates it privately and records the
results. See docs/plan/05-advisor-mode.md, "Phase 4 design".

private/campaign/ holds the truth: problem.toml, candidates.csv, data.csv and
decisions.md as of the last evaluated block. The workspace copy is rewritten
from it before every call, and everything read back from the workspace is
checked against it.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
import uuid

from ..advisor.blocks import block_folder, has_decision, open_block_number
from ..advisor.data import DataError, Table, format_number, load_campaign, read_table, standard_columns, write_table
from ..advisor.problem import ProblemError, load_problem
from .evaluator import Evaluator
from .execution import permissions, run
from .preflight import REPLAY_PROBE, check
from .runner import (SERVER, events_of, forbidden_paths, launch, only_workspace_tools, provenance,
                     readable_roots, runtime_info, runtime_view, tool_calls, write_json)

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
# What the runner records about each advisor call, summed over a block's attempts.
USAGE = ('turns', 'tool_calls', 'seconds', 'cost_usd',
         'input_tokens', 'cache_write_tokens', 'cache_read_tokens', 'output_tokens')
# Files of a public campaign the runner reads back, and their size limits.
LIMITS = {'data.csv': 20_000_000, 'decisions.md': 1_000_000, 'block file': 1_000_000}
# Block files that are outputs of propose.py, not inputs to reproducing it.
OUTPUTS = ('proposals.csv', 'propose.log')
OPT_RUN = '''"""Advisor mode: manage the campaign folder. See --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from agentoptimizer.advisor.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
'''


class ReplayError(RuntimeError):
    """The advisor's output broke a replay rule; the campaign stops here."""


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


def problem_toml(task, budget, block_size, seed):
    """An anonymous campaign for the task: x0..x{d-1}, minimize y."""
    indexed = getattr(task, 'indexed_pool', False)
    notes = ['An anonymous black-box objective. The names x0, x1, ... carry no meaning, and nothing '
             'is known about the function beyond the results in data.csv. Each block is evaluated '
             'in full before the next decision.']
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
    lines = ['name = "replay"', 'description = """', *notes, '"""', f'noise = "{noise}"', '',
             '[objective]', 'name = "y"', 'direction = "minimize"', '',
             '[budget]', f'total = {budget}', f'block_size = {block_size}']
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


def build_campaign(root, task, budget, block_size, seed):
    """workspace/ and private/ for a replay, with the empty campaign in both."""
    if task.n_constraints:
        raise ValueError('Replay does not support constrained tasks yet (phase 5).')
    workspace, private = Path(root) / 'workspace', Path(root) / 'private'
    for folder in [workspace / 'campaign', workspace / 'work', workspace / '.tmp', workspace / '.home',
                   private / 'campaign', private / 'blocks']:
        folder.mkdir(parents=True, exist_ok=True)
    install_advisor(workspace)
    truth = private / 'campaign'
    (truth / 'problem.toml').write_text(problem_toml(task, budget, block_size, seed))
    if getattr(task, 'indexed_pool', False):
        (truth / 'candidates.csv').write_text(candidates_csv(task))
    write_table(truth, Table(standard_columns(load_problem(truth))))
    (truth / 'decisions.md').write_text('')
    return workspace, private


def label_for(block, method):
    """The evaluator's optimizer label for a row: block number and method."""
    return re.sub(r'[^A-Za-z0-9_.+ -]', '_', f'block {block} {method}')[:100]


class Replay:
    """One replay campaign. `advise(block, attempt, logs)` makes one advisor call
    and returns its usage; `sandbox(command)` runs a shell command in the
    workspace under the agent's isolation and returns a CompletedProcess."""

    def __init__(self, workspace, private, evaluator, advise, sandbox,
                 python='.venv/bin/python', retries=1, campaign='run'):
        self.workspace = Path(workspace).resolve()
        self.private = Path(private).resolve()
        self.public = self.workspace / 'campaign'
        self.truth = self.private / 'campaign'
        self.evaluator, self.advise, self.sandbox = evaluator, advise, sandbox
        self.python, self.retries, self.campaign = python, retries, campaign
        self.problem = load_problem(self.truth)

    # Files in the workspace. The advisor could have replaced any of them with
    # a link, so the runner never follows one and caps what it reads.
    def _check_path(self, path):
        for part in [path, *path.parents]:
            if part == self.workspace:
                return
            if part.is_symlink():
                raise ReplayError(f'{part.relative_to(self.workspace)} is a link')
        raise ReplayError(f'{path} is outside the workspace')

    def read_public(self, path, limit):
        self._check_path(path)
        if not path.exists():
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ReplayError(f'{path.relative_to(self.workspace)} is not a regular file')
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ReplayError(f'{path.relative_to(self.workspace)} is larger than {limit} bytes')
        return data.decode()

    def write_public(self, path, text):
        self._check_path(path.parent)
        temporary = path.with_name(f'.{uuid.uuid4().hex}.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)

    def remaining(self):
        return self.problem.total_budget - len(read_table(self.truth, self.problem).rows)

    def run(self):
        """Advise, check and evaluate blocks until the budget is used or a block fails."""
        records = []
        while self.remaining() > 0:
            number = open_block_number(load_campaign(self.truth)[2])
            records.append(self.block(number))
            self.write_records(records)
            if records[-1]['status'] != 'evaluated':
                break
        return records

    def publish(self):
        for name in ['problem.toml', 'candidates.csv', 'data.csv']:
            if (self.truth / name).exists():
                self.write_public(self.public / name, (self.truth / name).read_text())

    def block(self, number):
        logs = self.private / 'blocks' / f'{number:03d}'
        record = {'block': number, 'attempts': 0, 'status': 'no proposals', 'rows': 0, 'methods': '',
                  'propose_sha256': '', **dict.fromkeys(USAGE, 0)}
        rows = []
        for attempt in range(1 + self.retries):
            self.publish()
            usage = self.advise(number, attempt, logs / f'attempt{attempt + 1}') or {}
            record['attempts'] += 1
            for key in USAGE:
                record[key] += usage.get(key, 0)
            rows, decisions = self.collect(number)
            if rows:
                break
        if not rows:
            return record
        script = self.reproduce(number, rows, decisions, logs)
        values = self.evaluate(number, rows, decisions)
        self.commit(rows, values, decisions)
        record.update(status='evaluated', rows=len(rows),
                      methods=' + '.join(dict.fromkeys(row['optimizer'] for row in rows)),
                      propose_sha256=hashlib.sha256(script.encode()).hexdigest())
        return record

    def collect(self, number):
        """The block's new pending rows and decisions.md, checked against the truth."""
        inbox = self.private / 'inbox'
        shutil.rmtree(inbox, ignore_errors=True)
        shutil.copytree(self.truth, inbox)
        text = self.read_public(self.public / 'data.csv', LIMITS['data.csv'])
        if text is None:
            raise ReplayError('campaign/data.csv was removed')
        (inbox / 'data.csv').write_text(text)
        try:
            _, table, _ = load_campaign(inbox)
        except (DataError, ProblemError) as exc:
            raise ReplayError(f'campaign/data.csv is not valid: {exc}') from None
        truth = read_table(self.truth, self.problem)
        if table.columns != truth.columns or table.rows[:len(truth.rows)] != truth.rows:
            raise ReplayError('campaign/data.csv changed rows or columns that were already recorded')
        rows = table.rows[len(truth.rows):]
        for row in rows:
            if (row['status'], row['block'], row['source']) != ('pending', str(number), 'advisor'):
                raise ReplayError(f'campaign/data.csv row {row["id"]} is not a pending advisor row of block '
                                  f'{number}; only accepted proposals may be added')
        if len(rows) > self.remaining():
            raise ReplayError(f'block {number} has {len(rows)} rows, over the remaining budget')
        decisions = self.read_public(self.public / 'decisions.md', LIMITS['decisions.md']) or ''
        if not decisions.startswith((self.truth / 'decisions.md').read_text()):
            raise ReplayError('decisions.md changed earlier entries; it is append-only')
        (inbox / 'decisions.md').write_text(decisions)
        if rows and not has_decision(inbox, number):
            raise ReplayError(f'decisions.md has no entry for block {number}')
        return rows, decisions

    def reproduce(self, number, rows, decisions, logs):
        """Rerun the block's propose.py in the sandbox on the data the block saw;
        its proposals must be the rows accepted. Returns the script."""
        source = block_folder(self.public, number)
        self._check_path(source)
        if not source.is_dir():
            raise ReplayError(f'campaign/blocks/{number:03d} is missing')
        files = {}
        for entry in sorted(source.iterdir()):
            if entry.is_file() and not entry.is_symlink():
                files[entry.name] = self.read_public(entry, LIMITS['block file'])
        if 'propose.py' not in files:
            raise ReplayError(f'campaign/blocks/{number:03d}/propose.py is missing')
        logs.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (logs / name).write_text(text)
        verify = self.workspace / '.tmp/verify'
        shutil.rmtree(verify, ignore_errors=True)
        folder = block_folder(verify / 'campaign', number)
        folder.mkdir(parents=True)
        for name in ['problem.toml', 'candidates.csv', 'data.csv']:
            if (self.truth / name).exists():
                shutil.copyfile(self.truth / name, verify / 'campaign' / name)
        (verify / 'campaign/decisions.md').write_text(decisions)
        for name, text in files.items():
            if name not in OUTPUTS:
                (folder / name).write_text(text)
        result = self.sandbox(f'{self.python} -I opt_run.py run .tmp/verify/campaign '
                              f'--block {number} --no-verify')
        (logs / 'reproduce.log').write_text(result.stdout + result.stderr)
        output = self.read_public(folder / 'proposals.csv', LIMITS['data.csv'])
        if result.returncode or output is None:
            raise ReplayError(f'block {number}: propose.py did not run again; see {logs / "reproduce.log"}')
        proposals = list(csv.DictReader(io.StringIO(output)))
        if [self.identity(row) for row in proposals] != [self.identity(row) for row in rows]:
            raise ReplayError(f'block {number}: the accepted rows are not what propose.py proposes '
                              'from the data the block saw')
        shutil.rmtree(verify)
        return files['propose.py']

    def identity(self, row):
        """What a row asks to evaluate: its candidate in a pool, else its values as accepted."""
        if self.problem.pool is not None:
            return row.get('candidate_id')
        return tuple(format_number(p, float(row[p.name])) for p in self.problem.parameters)

    def request(self, op, **kwargs):
        reply = self.evaluator.handle(self.campaign, {'request_id': uuid.uuid4().hex, 'op': op, **kwargs})
        if not reply['ok']:
            raise ReplayError(f'evaluator refused {op}: {reply["error"]}')
        return reply['result']

    def evaluate(self, number, rows, decisions):
        """Evaluate the rows in order, journaling the decisions with the first."""
        values, label, documents = [], None, {'documents': {'decisions.md': decisions}}
        for row in rows:
            if label_for(number, row['optimizer']) != label:
                label = label_for(number, row['optimizer'])
                self.request('label', name=label, **documents)
                documents = {}
            if self.problem.pool is not None:
                y = self.request('evaluate', candidate_index=int(row['candidate_id']))
            else:
                y = self.request('evaluate', x=[float(row[name]) for name in self.problem.names])
            values.append(float(y))
        return values

    def commit(self, rows, values, decisions):
        table = read_table(self.truth, self.problem)
        for row, value in zip(rows, values):
            table.rows.append({**row, 'status': 'done', self.problem.objective: repr(value)})
        write_table(self.truth, table)
        (self.truth / 'decisions.md').write_text(decisions)

    def write_records(self, records):
        with (self.private / 'blocks.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)


def system_prompt(agent):
    """The advisor's instructions: the policy, then the headless procedure."""
    return ((REPO / 'prompts/advisor.md').read_text() + '\n'
            + (REPO / 'prompts/advisor_headless.md').read_text()
            + f'\nAll commands and file access use the {SERVER[agent]} MCP tools.\n')


def usage_of(events):
    """Turns, cost and tokens of one CLI call, from its event stream (Claude or Codex)."""
    usage = dict.fromkeys(USAGE, 0)
    for event in events:
        if event.get('type') == 'result':  # Claude: one per call
            tokens = event.get('usage', {})
            usage.update(turns=event.get('num_turns', 0), cost_usd=event.get('total_cost_usd', 0.),
                         input_tokens=tokens.get('input_tokens', 0),
                         cache_write_tokens=tokens.get('cache_creation_input_tokens', 0),
                         cache_read_tokens=tokens.get('cache_read_input_tokens', 0),
                         output_tokens=tokens.get('output_tokens', 0))
        elif event.get('type') == 'turn.completed':  # Codex: input tokens include cached ones
            tokens = event.get('usage', {})
            cached = tokens.get('cached_input_tokens', 0)
            usage['turns'] += 1
            usage['input_tokens'] += tokens.get('input_tokens', 0) - cached
            usage['cache_read_tokens'] += cached
            usage['output_tokens'] += tokens.get('output_tokens', 0)
    return usage


def cli_advisor(args, workspace, private, policy):
    """advise() for Replay: one fresh headless CLI call per attempt."""
    system = system_prompt(args.agent)

    def advise(block, attempt, logs):
        prompt = f'Propose block {block} of the campaign in campaign/, following your instructions.'
        start = time.monotonic()
        launch(args, workspace, private, policy, prompt, system, logs)
        events = events_of(logs)
        tools = tool_calls(events)
        if not only_workspace_tools(tools, args.agent):
            raise ReplayError(f'block {block}: the advisor used a tool outside the sandbox; see {logs}')
        return {**usage_of(events), 'tool_calls': len(tools), 'seconds': round(time.monotonic() - start, 1)}
    return advise


def audit(evaluator, replay, records, error):
    """The journal and the private campaign agree; the run is complete or says why not."""
    journal = evaluator.db.execute("SELECT y FROM evaluations WHERE campaign='run' ORDER BY idx").fetchall()
    rows = read_table(replay.truth, replay.problem).rows
    report = {**evaluator.status('run'),
              'complete': evaluator.count('run') == evaluator.suite['budget'],
              'stopped': error or (records[-1]['status'] if records and records[-1]['status'] != 'evaluated' else ''),
              'unknown_evaluations': sum(y is None for (y,) in journal),
              'data_matches_journal': [float(r[replay.problem.objective]) for r in rows] == [y for (y,) in journal],
              'blocks': len(records), **{key: sum(r[key] for r in records) for key in USAGE}}
    report.pop('available_indices', None)
    write_json(replay.private / 'integrity_audit.json', report)
    if report['unknown_evaluations'] or not report['data_matches_journal']:
        raise RuntimeError('Integrity audit failed; see ' + str(replay.private / 'integrity_audit.json'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Replay advisor mode blind: a fresh advisor call per block.')
    parser.add_argument('--task', required=True, help='A key of src/agentoptimizer/tasks/registry.py')
    parser.add_argument('--budget', required=True, type=int)
    parser.add_argument('--block-size', type=int, default=8,
                        help='Preferred block size written to problem.toml (default 8); the advisor chooses')
    parser.add_argument('--seed', type=int, default=0, help='Campaign seed and private observation-noise seed')
    parser.add_argument('--agent', default='claude', choices=['codex', 'claude'], help='CLI that drives the model')
    parser.add_argument('--model', help='Default: gpt-6-astra (codex), claude-opus-5-5 (claude)')
    parser.add_argument('--effort', default='max', choices=['low', 'medium', 'high', 'xhigh', 'max'],
                        help='Reasoning effort (default: max, as in the benchmark).')
    parser.add_argument('--retries', type=int, default=1, help='Fresh calls for a block that accepted nothing')
    parser.add_argument('--agent-python', type=Path, default=REPO / '.agent-venv/bin/python')
    parser.add_argument('--codex', type=Path, default=shutil.which('codex'),
                        help='Codex CLI; also the per-command sandbox for both agents')
    parser.add_argument('--claude', type=Path, default=shutil.which('claude'))
    parser.add_argument('--runs-dir', type=Path, default=REPO / 'runs')
    parser.add_argument('--check-only', action='store_true', help='Validate isolation with no model call')
    args = parser.parse_args(argv)
    if sys.platform not in ('darwin', 'linux'):
        parser.error('The native executor currently supports macOS and Linux only.')
    if args.budget < 1 or args.block_size < 1 or args.seed < 0 or args.retries < 0 or args.codex is None:
        parser.error('Use a positive budget and block size, nonnegative seed and retries, and install Codex CLI.')
    if args.agent == 'claude' and args.claude is None:
        parser.error('Install and sign in to Claude Code CLI, or pass --claude.')
    args.model = args.model or {'codex': 'gpt-6-astra', 'claude': 'claude-opus-5-5'}[args.agent]
    # Preserve the venv executable path: resolving its symlink would bypass the venv.
    args.agent_python = args.agent_python.expanduser().absolute()
    args.codex = args.codex.expanduser().absolute()
    if args.claude is not None:
        args.claude = args.claude.expanduser().absolute()
    from ..tasks.registry import TASKS
    if args.task not in TASKS:
        parser.error('Unknown task. See src/agentoptimizer/tasks/registry.py.')
    task = TASKS[args.task]()
    if task.n_constraints:
        parser.error('Replay does not support constrained tasks yet.')
    if task.pool and args.budget > task.n_candidates:
        parser.error(f'A pool task allows at most {task.n_candidates} evaluations.')
    task.reseed(args.seed)
    runtime = runtime_info(args.agent_python)

    # An opaque basename keeps the task identity out of paths visible to the model.
    root = args.runs_dir.resolve() / ('replay-' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, mode=0o700)
    (root / 'private').mkdir(mode=0o700)
    workspace, private = build_campaign(root, task, args.budget, min(args.block_size, args.budget), args.seed)
    (workspace / 'AGENTS.md').write_text(system_prompt(args.agent))
    runtime_view(workspace, runtime)
    policy = {'codex': str(args.codex), 'python': str(args.agent_python), 'workspace': str(workspace),
              'permissions': permissions(workspace, readable_roots(args, runtime),
                                         writable=('campaign', 'work', '.tmp'), read_only=(), marker='AGENTS.md')}
    write_json(private / 'execution.json', policy)
    suite = {'budget': args.budget, 'seed': args.seed, 'campaigns': [{'id': 'run', 'task': args.task}],
             'documents': {'decisions.md': 'append'}}
    evaluator = Evaluator(suite, private / 'evaluations.sqlite', lambda _: task)
    write_json(private / 'scoring.json', {'task': args.task, 'task_name': task.name, 'f_opt': float(task.f_opt),
                                          'noisy': task.noisy, 'n_constraints': task.n_constraints,
                                          'seed': args.seed})
    replay = Replay(workspace, private, evaluator, cli_advisor(args, workspace, private, policy),
                    lambda command: run(policy, command, timeout=900), retries=args.retries)
    replay.publish()
    write_json(private / 'manifest.json', {
        'mode': 'replay', 'task': args.task, 'budget': args.budget, 'block_size': args.block_size,
        'seed': args.seed, 'agent': args.agent, 'model': args.model, 'reasoning_effort': args.effort,
        'retries': args.retries,
        **provenance(args, task, runtime, workspace, ['blind_runner', 'tasks', 'advisor', 'batch'])})
    print('Run directory: ' + str(root), flush=True)
    records, error = [], ''
    try:
        report = check(policy, forbidden_paths(task, root, private), private, REPLAY_PROBE)
        print(f"Isolation passed: {len(report['checks'])} checks, 0/{args.budget} evaluations.", flush=True)
        if not args.check_only:
            try:
                records = replay.run()
            except ReplayError as exc:
                error = str(exc)
                print('Replay stopped: ' + error, flush=True)
    finally:
        if sys.exc_info()[0] is None:
            report = audit(evaluator, replay, records, error)
            (private / 'scored_trace.csv').write_text(evaluator.scored_trace('run'))
        evaluator.db.close()
    print(f"Finished: {report['evaluations_used']}/{args.budget} evaluations in {report['blocks']} blocks"
          f" (${report['cost_usd']:.2f}). Campaign: {workspace / 'campaign'}")
    return 0 if report['complete'] else 1
