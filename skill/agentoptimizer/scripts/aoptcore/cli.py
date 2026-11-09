"""aopt: the commands the skill (and its user) run.

    aopt init [--import data.csv]     create .agentoptimizer/ around problem.yaml
    aopt status                       budget, best point, the current block and what to measure
    aopt validate                     check agent/proposal.json without committing it
    aopt commit                       record agent/proposal.json and agent/decision.md as a new block
    aopt suggest                      tell the block's optimizer new results; request the next point(s)
    aopt encode '{"name": value}'     a configuration as the optimizer's encoded point
    aopt decode '[...]'               an encoded point as a configuration

Every command takes --project (default: ./.agentoptimizer).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import driver, project as project_mod
from .data import DataError
from .problem import ProblemError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aopt", description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", type=Path, default=Path(".agentoptimizer"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="create the project folder around problem.yaml")
    p.add_argument("--import", dest="import_csv", type=Path, help="a CSV of earlier measurements, columns by name")
    sub.add_parser("status", help="budget, best point, current block, pending points")
    sub.add_parser("validate", help="check agent/proposal.json")
    sub.add_parser("commit", help="record the agent's proposal as a new block")
    sub.add_parser("suggest", help="pass back new results and request the next point(s)")
    p = sub.add_parser("encode", help="configuration JSON -> encoded point")
    p.add_argument("config")
    p = sub.add_parser("decode", help="encoded point JSON -> configuration")
    p.add_argument("point")
    a = parser.parse_args(argv)
    try:
        return COMMANDS[a.command](a)
    except (ProblemError, DataError, project_mod.Rejected, driver.HandleError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def cmd_init(a) -> int:
    proj = project_mod.init(a.project, a.import_csv)
    print(f"Initialized {proj.root}: {len(proj.problem.variables)} variables "
          f"({proj.encoder.dim} encoded coordinates), {proj.table.used} measurements, "
          f"budget {proj.problem.budget}.")
    return 0


def cmd_status(a) -> int:
    proj = project_mod.Project(a.project)
    t, prob = proj.table, proj.problem
    print(f"Problem: {prob.name}. {t.used} of {prob.budget} measurements used; {proj.remaining} remaining.")
    best = t.best()
    if best:
        config = ", ".join(f"{v.name}={best[v.name]}" for v in prob.variables)
        print(f"Best {prob.objective} ({'max' if prob.maximize else 'min'}imized): "
              f"{best[prob.objective]} at row {best['id']} ({config}).")
    else:
        print("No feasible measurement yet.")
    block = proj.current()
    if block is None:
        print("No block yet: the agent proposes the first one.")
        return 0
    progress = proj.block_progress(block)
    done = len(progress["rows"]) - len(progress["pending"])
    print(f"Block {block['block']}: {block['script']} as \"{block['label']}\" ({block['state']}), "
          f"{done} of {block['n_points']} measured, {len(progress['pending'])} pending.")
    if progress["pending"]:
        print("Measure the pending rows of data.csv, fill in the values, then run `aopt suggest`:")
        _show(proj, progress["pending"])
    elif progress["complete"]:
        print("The block is complete: ask the agent for the next block.")
    else:
        print("Run `aopt suggest` for the next point.")
    return 0


def cmd_validate(a) -> int:
    proj = project_mod.Project(a.project)
    proposal, _ = proj.read_agent_files()
    p = proj.validate(proposal, cancelling=len(proj.table.by_status("pending")))
    print(f"Valid: {p['n_points']} points of {p['script']} as \"{p['label']}\" ({p['state']}).")
    return 0


def cmd_commit(a) -> int:
    proj = project_mod.Project(a.project)
    proposal, decision = proj.read_agent_files()
    record = proj.commit(proposal, decision, settle=driver.settle)
    print(f"Committed block {record['block']}: {record['n_points']} points of {record['script']} "
          f"as \"{record['label']}\". Run `aopt suggest` for its first point(s).")
    return 0


def cmd_suggest(a) -> int:
    proj = project_mod.Project(a.project)
    rows, messages = driver.suggest(proj)
    for m in messages:
        print(m)
    block = proj.current()
    progress = proj.block_progress(block)
    if rows:
        print(f"Requested {len(rows)} point(s) (block {block['block']}, {block['label']}), now pending in data.csv:")
        _show(proj, rows)
        print(f"Measure them, fill in {proj.problem.objective}"
              + (f" and {', '.join(proj.problem.constraints)}" if proj.problem.constraints else "")
              + ", then run `aopt suggest` again.")
    elif progress["pending"]:
        print("Still waiting for these pending rows:")
        _show(proj, progress["pending"])
    elif proj.remaining <= 0:
        print("The budget is spent. The best point is in `aopt status`.")
    elif progress["complete"]:
        print(f"Block {block['block']} is complete: ask the agent for the next block.")
    return 0


def cmd_encode(a) -> int:
    proj = project_mod.Project(a.project)
    print(json.dumps(proj.encoder.encode(json.loads(a.config))))
    return 0


def cmd_decode(a) -> int:
    proj = project_mod.Project(a.project)
    print(json.dumps(proj.encoder.decode(json.loads(a.point))))
    return 0


def _show(proj, rows) -> None:
    names = ["id"] + [v.name for v in proj.problem.variables]
    print("  " + "  ".join(names) + ("  replicate_of" if any(r["replicate_of"] for r in rows) else ""))
    for r in rows:
        print("  " + "  ".join(str(r[n]) for n in names) + (f"  {r['replicate_of']}" if r["replicate_of"] else ""))


COMMANDS = {"init": cmd_init, "status": cmd_status, "validate": cmd_validate, "commit": cmd_commit,
            "suggest": cmd_suggest, "encode": cmd_encode, "decode": cmd_decode}
