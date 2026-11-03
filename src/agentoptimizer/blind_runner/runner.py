"""Run one blind campaign: prepare the workspace, pass the isolation preflight,
then run the loop with a headless advisor until the budget is used.

Trusted files live in private/; only workspace/ and an optimizer-only runtime
are exposed to agent commands. No VM, socket API, or shared campaign state.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from ..advisor.data import read_table
from .agent_cli import cli_advisor, system_prompt, write_json
from .campaign import build_campaign
from .evaluator import Evaluator
from .execution import permissions, run
from .loop import USAGE, Loop, LoopError
from .preflight import check

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]


def runtime_info(python):
    query = """import sys,json,importlib.util,importlib.metadata as m
names=['numpy','scipy','cma','botorch','gpytorch','torch','Py-BOBYQA','pandas','matplotlib']
print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,
 'executable':sys.executable,'version':sys.version,
 'backends':[n for n in ['ioh','cocoex','opfunu','pymoo','yahpo_gym','agentoptimizer']
   if importlib.util.find_spec(n) is not None],
 'packages':{n:m.version(n) for n in names}}))
"""
    info = json.loads(subprocess.check_output([str(python), '-I', '-c', query], text=True))
    if info['backends']:
        raise ValueError('Agent runtime contains objective backends: ' + ', '.join(info['backends'])
                         + '. Create it with: UV_PROJECT_ENVIRONMENT=.agent-venv uv sync --only-group agent')
    return info


def runtime_view(workspace, runtime):
    # A read-only venv view removes installation aliases (e.g. uv's versionless
    # symlink) without granting access to their parent directories on macOS.
    venv = workspace / '.venv'
    (venv / 'bin').mkdir(parents=True)
    executable = Path(runtime['executable']).resolve()
    (venv / 'bin/python').symlink_to(executable)
    (venv / 'pyvenv.cfg').write_text(
        f'home = {executable.parent}\ninclude-system-site-packages = false\n')
    (venv / 'lib').symlink_to(Path(runtime['prefix']).resolve() / 'lib', target_is_directory=True)
    if (Path(runtime['prefix']) / 'lib64').exists():
        (venv / 'lib64').symlink_to('lib', target_is_directory=True)


def readable_roots(args, runtime):
    roots = {runtime['prefix'], runtime['base_prefix'],
             str(Path(runtime['prefix']).resolve()), str(Path(runtime['base_prefix']).resolve())}
    if sys.platform.startswith('linux'):
        # Codex's Linux sandbox re-executes the codex binary inside bubblewrap, so its
        # package (bin/, codex-resources/, codex-path/; no credentials) must be readable.
        roots.add(str(Path(args.codex).resolve().parents[1]))
    return sorted(roots)


def provenance(args, task, runtime, workspace, folders):
    """What a run was made from: source commit, CLI and runtime versions, hashes."""
    return {
        'source_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_status': subprocess.check_output(['git', '-C', str(REPO), 'status', '--short'], text=True),
        'codex_version': subprocess.check_output([str(args.codex), '--version'], text=True).strip(),
        **({'claude_version': subprocess.check_output([str(args.claude), '--version'], text=True).strip()}
           if args.agent == 'claude' else {}),
        'agent_runtime': runtime,
        'private_data_sha256': {str(Path(p).relative_to(REPO)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in getattr(task, 'private_data_paths', ())},
        'source_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in folders for p in (PACKAGE / folder).glob('*.py')},
        'public_sha256': {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in workspace.rglob('*') if p.is_file() and not p.is_symlink()},
    }


def forbidden_paths(task, root, private):
    """Paths the preflight must fail to read from inside the sandbox."""
    (private / 'canary.txt').write_text('Private evaluator data: access must fail.')
    (root / 'sibling_canary.txt').write_text('Another campaign: access must fail.')
    return [private / 'canary.txt', root / 'sibling_canary.txt', REPO / 'README.md',
            Path(inspect.getfile(type(task))), private / 'evaluations.sqlite',
            Path.home() / '.codex/auth.json', Path.home() / '.claude.json', Path(__file__),
            *getattr(task, 'private_data_paths', ())]


def audit(evaluator, loop, records, error, check_only=False):
    """The journal and the private campaign agree; the run is complete or says why not."""
    journal = evaluator.db.execute("SELECT y FROM evaluations WHERE campaign='run' ORDER BY idx").fetchall()
    rows = read_table(loop.truth, loop.problem).rows
    report = {**evaluator.status('run'),
              'complete': evaluator.count('run') == evaluator.suite['budget'],
              'stopped': error or (records[-1]['status'] if records and records[-1]['status'] != 'evaluated' else ''),
              'unknown_evaluations': sum(y is None for (y,) in journal),
              'data_matches_journal': [float(r[loop.problem.objective]) for r in rows] == [y for (y,) in journal],
              'blocks': len(records), 'check_only': check_only,
              **{key: sum(r[key] for r in records) for key in USAGE}}
    write_json(loop.private / 'integrity_audit.json', report)
    if report['unknown_evaluations'] or not report['data_matches_journal']:
        raise RuntimeError('Integrity audit failed; see ' + str(loop.private / 'integrity_audit.json'))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run one blind campaign: a fresh headless advisor call per block.')
    parser.add_argument('--task', required=True, help='A key of src/agentoptimizer/tasks/registry.py')
    parser.add_argument('--budget', required=True, type=int)
    parser.add_argument('--block-size', type=int,
                        help='Preferred block size written to problem.toml (default 10%% of the budget); '
                             'the advisor chooses')
    parser.add_argument('--round-size', type=int, default=1,
                        help='Evaluations per round before results come back (default 1: sequential)')
    parser.add_argument('--seed', type=int, default=0, help='Campaign seed and private observation-noise seed')
    parser.add_argument('--agent', default='claude', choices=['codex', 'claude'], help='CLI that drives the model')
    parser.add_argument('--model', help='Default: gpt-6-astra (codex), claude-opus-5-5 (claude)')
    parser.add_argument('--effort', default='medium', choices=['low', 'medium', 'high', 'xhigh', 'max'],
                        help='Reasoning effort (default: medium; the legacy live-session agent used max).')
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
    if args.block_size is None:
        args.block_size = max(1, round(args.budget / 10))
    if min(args.budget, args.block_size, args.round_size) < 1 or args.seed < 0 or args.retries < 0 \
            or args.codex is None:
        parser.error('Use a positive budget, block and round size, nonnegative seed and retries, '
                     'and install Codex CLI.')
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
        parser.error('Constrained tasks are not supported yet (docs/plan/06-unified-loop.md).')
    if task.pool and args.budget > task.n_candidates:
        parser.error(f'A pool task allows at most {task.n_candidates} evaluations.')
    task.reseed(args.seed)
    runtime = runtime_info(args.agent_python)

    # An opaque basename keeps the task identity out of paths visible to the model.
    root = args.runs_dir.resolve() / ('run-' + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, mode=0o700)
    (root / 'private').mkdir(mode=0o700)
    workspace, private = build_campaign(root, task, args.budget, min(args.block_size, args.budget),
                                        args.round_size, args.seed)
    (workspace / 'AGENTS.md').write_text(system_prompt(args.agent))
    runtime_view(workspace, runtime)
    policy = {'codex': str(args.codex), 'python': str(args.agent_python), 'workspace': str(workspace),
              'permissions': permissions(workspace, readable_roots(args, runtime))}
    write_json(private / 'execution.json', policy)
    suite = {'budget': args.budget, 'seed': args.seed, 'campaigns': [{'id': 'run', 'task': args.task}]}
    evaluator = Evaluator(suite, private / 'evaluations.sqlite', lambda _: task)
    write_json(private / 'scoring.json', {'task': args.task, 'task_name': task.name, 'f_opt': float(task.f_opt),
                                          'noisy': task.noisy, 'n_constraints': task.n_constraints,
                                          'seed': args.seed})
    loop = Loop(workspace, private, evaluator, cli_advisor(args, workspace, private, policy),
                lambda command: run(policy, command, timeout=900), retries=args.retries)
    loop.publish()
    write_json(private / 'manifest.json', {
        'task': args.task, 'budget': args.budget, 'block_size': args.block_size,
        'round_size': args.round_size,
        'seed': args.seed, 'agent': args.agent, 'model': args.model, 'reasoning_effort': args.effort,
        'retries': args.retries,
        **provenance(args, task, runtime, workspace, ['blind_runner', 'tasks', 'advisor', 'batch'])})
    print('Run directory: ' + str(root), flush=True)
    error = ''
    try:
        report = check(policy, forbidden_paths(task, root, private), private)
        print(f"Isolation passed: {len(report['checks'])} checks, 0/{args.budget} evaluations.", flush=True)
        if not args.check_only:
            try:
                loop.run()
            except LoopError as exc:
                error = str(exc)
                print('Stopped: ' + error, flush=True)
    finally:
        if sys.exc_info()[0] is None:
            report = audit(evaluator, loop, loop.records, error, args.check_only)
            (private / 'scored_trace.csv').write_text(evaluator.scored_trace('run'))
        evaluator.db.close()
    print(f"Finished: {report['evaluations_used']}/{args.budget} evaluations in {report['blocks']} blocks"
          f" (${report['cost_usd']:.2f}). Campaign: {workspace / 'campaign'}")
    return 0 if report['complete'] else 1
