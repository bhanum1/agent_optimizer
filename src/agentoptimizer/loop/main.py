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

from . import agents, campaign, preflight, results, sandbox

REPO = campaign.REPO


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentoptimizer.loop",
                                     description="Run an agent campaign, one skill call per block.")
    parser.add_argument("--task", required=True, help="A key of src/agentoptimizer/tasks/registry.py")
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", default="claude", choices=["codex", "claude"])
    parser.add_argument("--model", help="Default: claude-opus-5-5 (claude), gpt-6-astra (codex)")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
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
    from agentoptimizer.tasks.registry import TASKS
    if args.task not in TASKS:
        parser.error("Unknown task. See src/agentoptimizer/tasks/registry.py.")
    task = TASKS[args.task]()
    if args.budget < 1 or (task.pool and args.budget > task.n_candidates):
        parser.error("Budget must be positive and at most the pool size.")
    task.reseed(args.seed)

    cfg = {"task": args.task, "agent": args.agent, "effort": args.effort, "call_timeout": args.call_timeout,
           "model": args.model or {"codex": "gpt-6-astra", "claude": "claude-opus-5-5"}[args.agent],
           "codex": str(args.codex.expanduser().absolute()),
           "claude": str(args.claude.expanduser().absolute()) if args.claude else None}
    runtime = sandbox.runtime_info(args.agent_python.expanduser().absolute())
    # An opaque name keeps the task identity out of paths the model sees.
    root = args.runs_dir.resolve() / ("run-" + uuid.uuid4().hex[:12])
    root.mkdir(parents=True, mode=0o700)
    run = campaign.Campaign.prepare(task, args.budget, args.seed, root)
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

    started = datetime.now(timezone.utc).isoformat()
    reason = run.run(call)
    report = results.audit(run, args.agent)
    results.write(run, cfg, reason, report, started)
    print(f"Finished ({reason}): {len(run.store)}/{args.budget} evaluations, audit passed. "
          f"Results: {root / 'results'}")
    return 0
