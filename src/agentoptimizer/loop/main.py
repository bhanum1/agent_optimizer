"""Command line: prepare a campaign, check isolation, run the loop."""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import agents, campaign, preflight, results, sandbox, user_data

REPO = campaign.REPO


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentoptimizer.loop",
                                     description="Run an agent campaign, one skill call per block.")
    parser.add_argument("--task", help="A key of src/agentoptimizer/tasks/registry.py")
    parser.add_argument("--budget", type=int, help="Evaluations in the campaign (with --task)")
    parser.add_argument("--data", type=Path, help="Propose mode: a CSV with columns x0,x1,...,y (minimized); "
                        "one skill call proposes the next block, nothing is evaluated")
    parser.add_argument("--bounds", help="With --data: lo,hi for every variable, or a JSON list of [lo, hi] "
                        "pairs (inline or a file)")
    parser.add_argument("--remaining", type=int, help="With --data: evaluations left after this data; "
                        "the block is at most this long")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", default="claude", choices=["codex", "claude"])
    parser.add_argument("--model", help="Default: claude-opus-5-5 (claude), gpt-6-astra (codex)")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--context", default="fresh", choices=["fresh", "persistent"],
                        help="fresh: a new CLI session per call; persistent: resume one session (claude only)")
    parser.add_argument("--call-timeout", type=int, default=3600, help="Seconds per skill call")
    parser.add_argument("--agent-python", type=Path, default=REPO / ".agent-venv/bin/python")
    parser.add_argument("--codex", type=Path, default=shutil.which("codex"),
                        help="Codex CLI; provides the sandbox for both agents")
    parser.add_argument("--claude", type=Path, default=shutil.which("claude"))
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--check-only", action="store_true", help="Check isolation; no model call")
    args = parser.parse_args(argv)
    if sys.platform not in ("darwin", "linux") or args.codex is None:
        parser.error("Needs macOS or Linux and the Codex CLI (for its sandbox).")
    if args.agent == "claude" and args.claude is None:
        parser.error("Install Claude Code CLI or pass --claude.")
    if args.data:
        if args.task or args.budget is not None or not args.bounds or not args.remaining or args.remaining < 1:
            parser.error("--data needs --bounds and a positive --remaining, and takes no --task or --budget.")
        try:
            task = user_data.UserData(args.data, args.bounds)
        except ValueError as e:
            parser.error(f"{args.data}: {e}")
        args.task, args.budget = task.name, len(task.y) + args.remaining
    else:
        from agentoptimizer.tasks.registry import TASKS
        if args.task not in TASKS or args.budget is None:
            parser.error("Give --task (see src/agentoptimizer/tasks/registry.py) and --budget, or --data.")
        task = TASKS[args.task]()
    if args.budget < 1 or (task.pool and args.budget > task.n_candidates):
        parser.error("Budget must be positive and at most the pool size.")
    task.reseed(args.seed)

    if args.context == "persistent" and args.agent != "claude":
        parser.error("--context persistent needs --agent claude.")
    cfg = {"task": args.task, "agent": args.agent, "effort": args.effort, "context": args.context, "call_timeout": args.call_timeout,
           "model": args.model or {"codex": "gpt-6-astra", "claude": "claude-opus-5-5"}[args.agent],
           "codex": str(args.codex.expanduser().absolute()),
           "claude": str(args.claude.expanduser().absolute()) if args.claude else None}
    runtime = sandbox.runtime_info(args.agent_python.expanduser().absolute())
    # An opaque name keeps the task identity out of paths the model sees.
    root = args.runs_dir.resolve() / ("run-" + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, mode=0o700)
    run = campaign.Campaign.prepare(task, args.budget, args.seed, root)
    if args.data:
        task.fill(run.store)
    sandbox.link_venv(run.workspace, runtime)
    roots = sandbox.runtime_roots(runtime, cfg["codex"])
    cfg["policy"] = {"codex": cfg["codex"], "workspace": str(run.workspace),
                     "agent": sandbox.permissions(run.workspace, roots, sandbox.AGENT_WRITES),
                     "host": sandbox.permissions(run.workspace, roots, sandbox.HOST_WRITES)}
    (run.private / "policy.json").write_text(json.dumps(cfg["policy"], indent=2))
    (run.private / "manifest.json").write_text(json.dumps(campaign.manifest(run, cfg, runtime), indent=2))
    run.prefix = {w: sandbox.prefix(cfg["policy"], w) for w in ("agent", "host")}
    print("Run directory: " + str(root), flush=True)

    (run.private / "canary.txt").write_text("private")
    system_tmp = Path(tempfile.mkdtemp(prefix="loop_canary_")) / "canary.txt"
    system_tmp.write_text("system temp")
    forbidden = [run.private / "canary.txt", system_tmp, run.private / "journal.csv", Path(inspect.getfile(type(task))),
                 REPO / "README.md", Path.home() / ".codex/auth.json", Path.home() / ".claude.json",
                 *getattr(task, "private_data_paths", ())]
    try:
        report = preflight.check(cfg["policy"], forbidden, run.private)
    finally:
        shutil.rmtree(system_tmp.parent)
    print(f"Isolation passed: {sum(map(len, report.values()))} checks.", flush=True)
    if args.check_only:
        return 0

    def call(number, prompt):
        code = agents.call(cfg, run.workspace, run.private, number, prompt)
        used = agents.tool_calls(run.private / "calls" / f"{number:03d}" / "events.jsonl", args.agent)
        unexpected = [t for s, t in used if s == "unexpected"]
        if unexpected:
            raise RuntimeError(f"Call {number} used tools outside the sandbox: {unexpected}")
        return code

    if args.data:
        return propose(run, call, root)

    started = datetime.now(timezone.utc).isoformat()
    reason = run.run(call)
    report = results.audit(run, args.agent)
    results.write(run, cfg, reason, report, started)
    print(f"Finished ({reason}): {len(run.store)}/{args.budget} evaluations, audit passed. "
          f"Results: {root / 'results'}")
    return 0


def propose(run, call, root: Path) -> int:
    """Propose mode: write the agent's block, its decision and its saved context to results/."""
    try:
        p = run.propose(call)
    except campaign.Rejected as e:
        print(f"No valid proposal: {e} See {run.decisions}.", file=sys.stderr)
        return 1
    out = root / "results"
    out.mkdir()
    shown = {k: v for k, v in p.items() if not k.startswith("_")}
    (out / "proposal.json").write_text(json.dumps(shown, indent=2) + "\n")
    (out / Path(p["script"]).name).write_text(p["_source"])  # the optimizer the block runs, as the agent left it
    shutil.copyfile(run.workspace / "agent" / "decision.md", out / "decision.md")
    shutil.copyfile(run.workspace / "agent" / "context.md", out / "context.md")
    print(f"Proposed {p['n_points']} evaluations of {p['script']} ({p['label']}, {p['state']}): {out}")
    return 0
