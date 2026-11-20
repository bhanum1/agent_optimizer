"""Counterfactual branches of a finished agent campaign.

At a decision point (after `after` measurements) the campaign is rewound and, instead of what the
agent chose, one of its earlier blocks is run on to the end of the budget: an optimizer the agent
had saved (`state: continue`) or one of its scripts started afresh on the data so far. Nothing new
is decided by a model; the branch is the agent's own code with the decision taken away.

The workspace is copied, data.csv cut to its first `after` rows, and the task's noise stream
replayed over those rows, so the branch sees the same noise draws the agent saw. Each repeat after
the first reseeds the noise from the branch point on, to show the spread.

    python -m aobench.counterfactual --runs benchmark/runs/opus_high --task noise_dominated --seed 4 \
        --after 464 --proposal '{"script": "work/lbo.py", "label": "lbo1", "args": {}, "state": "continue"}' \
        --name keep_local_gp --repeats 3 --out benchmark/runs/counterfactual
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .agent.campaign import Campaign


def workspace(runs: Path, task: str, seed: int) -> Path:
    """A campaign's workspace: where workspace.txt says, or the run folder's own copy of it
    (runs made on another machine and copied here keep the other machine's path)."""
    recorded = Path((runs / task / "agent" / f"seed_{seed}" / "workspace.txt").read_text().strip())
    return recorded if recorded.is_dir() else runs / "workspaces" / recorded.name


def branch(runs: Path, task: str, seed: int, after: int, proposal: dict, out: Path, repeat: int) -> Path:
    source = workspace(runs, task, seed)
    folder = out / task / f"seed_{seed}" / f"{proposal['label']}_{after}" / f"r{repeat}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    work = folder / "workspace"
    shutil.copytree(source, work, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    project = work / ".agentoptimizer"

    with open(project / "data.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    kept = rows[:after]
    last_block = int(kept[-1]["block"])
    with open(project / "data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    # Later blocks never happened in the branch. Saved optimizers from them go too.
    for spec in (project / "blocks").glob("*.json"):
        if json.loads(spec.read_text()).get("block", 0) > last_block:
            spec.unlink()
            spec.with_suffix(".py").unlink(missing_ok=True)
    for spec in (project / "state").glob("*.json"):
        if json.loads(spec.read_text()).get("block", 0) > last_block:
            spec.unlink()
            spec.with_suffix(".state").unlink(missing_ok=True)

    c = Campaign.__new__(Campaign)
    Campaign.__init__(c, task, seed, folder / "logs_root", folder / "tmp", agent="fake", model=None,
                      effort=None, call_timeout=0, max_calls=0)
    c.workspace, c.project, c.sandboxed = work, project, False
    c.logs = folder
    c.result = folder / "result.csv"
    for r in kept:  # replay: the same noise draws, and the harness's own record
        ev = c.task.evaluate(np.array([float(r[f"x{i}"]) for i in range(c.task.dim)]))
        if not np.isclose(float(r["y"]), ev.y, rtol=1e-9, atol=0):
            raise RuntimeError(f"row {r['id']} does not match its replay")
        if ev.y < c.best_y:
            c.best_y, c.regret = ev.y, ev.y_true - c.task.f_opt
        c.rows.append((len(c.rows) + 1, ev.y, c.regret))
    if repeat:
        c.task._noise = np.random.default_rng([seed, after, repeat, 7919])

    if proposal.get("state") == "continue":
        # Continue the optimizer as it ran then: the agent may have edited its script since.
        state = json.loads((project / "state" / f"{proposal['label']}.json").read_text())
        shutil.copyfile(project / state["source_file"], project / proposal["script"])

    remaining = c.budget - after
    if "points" in proposal:  # a list of points: the branch measures them, up to the budget
        proposal = {**proposal, "points": proposal["points"][:remaining]}
    else:
        proposal = {"warm_start": True, **proposal, "n_points": remaining}
    (project / "agent" / "proposal.json").write_text(json.dumps(proposal))
    c.aopt("commit")
    used, messages = c.run_block()
    (folder / "messages.txt").write_text("\n".join(messages))
    c.finish("branch")
    print(f"{proposal['label']}@{after} r{repeat}: {used} evaluations, final log10 regret "
          f"{np.log10(max(c.regret, 1e-8)):.2f}", flush=True)
    return c.result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--after", type=int, required=True)
    p.add_argument("--proposal", required=True,
                   help="the block to run instead, as aopt proposal JSON, or @file to read it from a file")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("benchmark/runs/counterfactual"))
    a = p.parse_args(argv)
    for r in range(a.repeats):
        branch(a.runs.resolve(), a.task, a.seed, a.after, json.loads(Path(a.proposal[1:]).read_text() if a.proposal.startswith("@") else a.proposal), a.out.resolve(), r)


if __name__ == "__main__":
    main()
