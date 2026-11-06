"""After a campaign: audit it, then write results/ in the baseline convention.

results/
  trace.csv     one row per evaluation, as baselines/campaign.py writes for
                baselines; y_best_so_far is the incumbent's noise-free value
  meta.json     task, budget, outcome, agent, and the audit
  decisions.md  copy of the decision record
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone

from . import agents

DECISION = re.compile(r"^## Decision after evaluation (\d+) \(call (\d+)\)\n\n(.*?)\n\nProposal: `([^`\n]*)`$",
                      re.MULTILINE | re.DOTALL)


def audit(run, agent: str | None) -> dict:
    """Check the campaign's records agree with each other. Raises if any check fails."""
    rows = run.store.rows
    with open(run.store.public, newline="") as f:
        public = list(csv.DictReader(f))
    decisions = run.decisions.read_text()
    found = [(int(a), int(c), json.loads(p)) for a, c, _, p in DECISION.findall(decisions)]
    blocks = [e for e in run.log if "block" in e]
    checks = {
        "within_budget": len(rows) <= run.budget,
        "data_matches_journal": public == [{c: r[c] for c in run.store.columns} for r in rows],
        "decisions_unmodified": decisions == run.written,
        "one_decision_per_proposal": found == [(e["after"], e["call"], e["proposal"]) for e in run.log],
        "decision_before_each_block": all(e["after"] == sum(b["used"] for b in blocks if b["block"] < e["block"])
                                          for e in blocks),
        "blocks_within_length": all(e["used"] <= e["proposal"]["n_points"] for e in blocks),
        "rows_match_blocks": [(int(r["block"]), r["optimizer"]) for r in rows]
                             == [(e["block"], e["proposal"]["label"]) for e in blocks for _ in range(e["used"])],
        "block_scripts_kept": all((run.workspace / "blocks" / e["script"]).is_file() for e in blocks),
    }
    tools = []
    if agent is not None:
        for events in sorted((run.private / "calls").glob("*/events.jsonl")):
            tools += agents.tool_calls(events, agent)
        checks["only_sandbox_tools"] = all(s != "unexpected" for s, _ in tools)
    report = {"passed": all(checks.values()), "checks": checks, "tool_calls": len(tools)}
    (run.private / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError(f"Audit failed: {failed}; see {run.private / 'audit.json'}")
    return report


def write(run, cfg: dict, reason: str, report: dict, started: str) -> None:
    task, out = run.task, run.root / "results"
    out.mkdir(exist_ok=True)
    columns = (["eval_idx", "block", "optimizer"] + (["candidate_index"] if run.store.indexed else [])
               + [f"x{d}" for d in range(task.dim)] + ["y"] + (["y_true"] if task.noisy else [])
               + ([f"g{i}" for i in range(task.n_constraints)] + ["feasible"] if task.n_constraints else [])
               + ["y_best_so_far", "eval_seconds", "cumulative_seconds"])
    best_y = best_score = best_x = best_index = None
    seconds = 0.0
    with open(out / "trace.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in run.store.rows:
            y, seconds = float(r["y"]), seconds + float(r["eval_seconds"])
            feasible = not task.n_constraints or r["feasible"] == "True"
            if feasible and (best_y is None or y < best_y):
                best_y = y
                best_score = float(r["y_true"]) if task.noisy else y
                best_x = [float(r[f"x{d}"]) for d in range(task.dim)]
                best_index = int(r["candidate_index"]) if run.store.indexed else None
            row = {**r, "eval_idx": r["eval"], "y_best_so_far": "" if best_score is None else best_score,
                   "cumulative_seconds": seconds}
            writer.writerow([row[c] for c in columns])
    meta = {
        "task": cfg["task"], "task_name": task.name, "task_dim": task.dim, "task_f_opt": float(task.f_opt),
        "task_noisy": task.noisy, "task_n_constraints": task.n_constraints,
        "budget": run.budget, "requested_budget": run.budget, "seed": run.seed,
        "evaluations_used": len(run.store.rows), "stop_reason": reason,
        "best_y": best_y, "best_y_true": best_score if task.noisy else None, "best_x": best_x,
        **({"best_candidate_index": best_index} if run.store.indexed else {}),
        "agent": cfg["agent"], "model": cfg["model"], "effort": cfg["effort"], "context": cfg.get("context", "fresh"),
        "skill_calls": len(list((run.private / "calls").glob("*"))), "blocks": run.blocks,
        "start_time_utc": started, "end_time_utc": datetime.now(timezone.utc).isoformat(),
        "audit": report,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    shutil.copyfile(run.decisions, out / "decisions.md")
