"""Counterfactual branches for the grey-box campaigns of Figure 4.

    .venv/bin/python -m aobench.graybox_branches --upstream ~/path/to/cheapest_lunch

A branch is Figure 3's counterfactual, on a pool campaign: rewind to a decision and let the
optimizer the agent was running carry on to the end of the budget, instead of the one it
switched to. What that needs is the pool's features and the block's own script, neither of
which is in this repository -- `benchmark/graybox/` holds each campaign's trace and one label
column per pool, nothing else. Both live in the collaborators' repository, which is not a
dependency of this one and may not be present, so this module is run by hand and commits only
its result: a few dozen rows of regret per campaign, under `benchmark/graybox/branches/`.
Figure 4 reads those, and redraws with no access to any of this.

The blocks are the agent's own scripts, kept as it wrote them (`agent_artifacts/blocks/`),
and every block ran `state: "restart", warm_start: true` -- constructed fresh, then told the
history -- so a branch needs no saved optimizer state, only the script, its arguments and the
observations up to the decision. The pool the agent saw is an anonymized subset of the task
table's columns: `columns()` recovers which, by matching each variable's bounds. `--check`
replays the block that did run from its own start, which must reproduce the campaign's picks
exactly; the run prints that check and refuses to write if it fails.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

# (task, seed) -> the campaign's decisions: (evaluations done, the block that was running,
# the block it was replaced by). The branch runs the first to the end of the budget.
CAMPAIGNS = {
    ("mat_5", 1): [(70, "004_gpei3", "005_gpei4"), (85, "005_gpei4", "006_knn6")],
}
METHOD = "cheapest_lunch_opus_5_5_high_domain_context"


def columns(table: np.ndarray, variables: list[dict]) -> list[int]:
    """Which column of the task table each anonymous variable is, by its bounds.

    The campaign's candidates.csv is not kept, but its problem.yaml records every variable's
    low and high, and those are the column's own min and max: enough to recover the subset
    and its order. A handful of columns share bounds, so they are taken in order, which the
    `--check` replay then confirms end to end.
    """
    lo, hi = table.min(0), table.max(0)
    used, out = set(), []
    for v in variables:
        m = [i for i in range(table.shape[1])
             if i not in used and np.isclose(lo[i], v["low"]) and np.isclose(hi[i], v["high"])]
        if not m:
            raise SystemExit(f"no column of the task table has {v['name']}'s bounds")
        out.append(m[0])
        used.add(m[0])
    return out


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def campaign(upstream: Path, task: str, seed: int):
    """The campaign's pool, its picks and the objective it was told, from the task table."""
    run = upstream / "results/tasks" / task / METHOD / f"seed_{seed}"
    prob = json.loads((run / "agent_artifacts/problem.yaml").read_text())
    with open(upstream / "data" / task / "main.csv") as f:
        head = next(csv.reader(f))
    keep = [i for i, c in enumerate(head) if c not in ("mol_id", "y")]
    table = np.loadtxt(upstream / "data" / task / "main.csv", delimiter=",", skiprows=1,
                       usecols=keep)
    y = np.loadtxt(upstream / "data" / task / "main.csv", delimiter=",", skiprows=1,
                   usecols=[head.index("y")])
    space = {"variables": prob["variables"], "candidates": table[:, columns(table, prob["variables"])]}
    rows = list(csv.DictReader(open(run / "trace.csv")))
    idx = np.array([int(r["candidate_index"]) for r in rows])
    obj = np.array([float(r["objective_value"]) for r in rows])
    if not np.allclose(obj, -y[idx]):
        raise SystemExit(f"{task} seed {seed}: the trace does not match the task table")
    return run, space, y, idx, obj


def run_block(run: Path, space, y, idx, obj, block: str, after: int, n: int):
    """Construct the block as the campaign did, tell it the first `after` evaluations, run on."""
    mod = load(run / f"agent_artifacts/blocks/{block}.py")
    args = json.loads((run / f"agent_artifacts/blocks/{block}.json").read_text())["args"]
    opt = mod.Optimizer(space, **args)
    opt.tell(idx[:after], obj[:after], [[]] * after)
    picks = []
    for _ in range(n):
        i = int(opt.ask(1)[0])
        opt.tell([i], [-y[i]], [[]])
        picks.append(i)
    return picks


def branches(upstream: Path, task: str, seed: int, check: bool = True):
    run, space, y, idx, obj = campaign(upstream, task, seed)
    best = y.max()  # every pool task here is a maximization written as minimize -y
    out = {}
    for after, running, replaced in CAMPAIGNS[(task, seed)]:
        if check:  # the block that did run, from its own start, must give the campaign's picks
            start = json.loads((run / f"agent_artifacts/blocks/{running}.json").read_text())["after"]
            n = after - start
            got = run_block(run, space, y, idx, obj, running, start, n)
            if got != [int(v) for v in idx[start:after]]:
                raise SystemExit(f"{running} does not replay {task} seed {seed} evals "
                                 f"{start + 1}-{after}; the branch would be fiction")
            print(f"  check: {running} replays evals {start + 1}-{after} exactly")
        picks = run_block(run, space, y, idx, obj, running, after, len(idx) - after)
        got = np.maximum.accumulate(np.concatenate([[y[idx[:after]].max()], y[picks]]))[1:]
        out[after] = best - got
        hit = next((after + j + 1 for j, v in enumerate(out[after]) if v == 0), None)
        print(f"  branch from {after}: {running} instead of {replaced}, "
              f"optimum at eval {hit}" if hit else f"  branch from {after}: never reaches it")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--upstream", type=Path, required=True,
                   help="the collaborators' repository, with data/ and results/")
    p.add_argument("--out", type=Path, default=Path("benchmark/graybox/branches"))
    p.add_argument("--no-check", action="store_true", help="skip the replay check (do not)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for task, seed in CAMPAIGNS:
        print(f"{task} seed {seed}:")
        out = branches(a.upstream.expanduser(), task, seed, check=not a.no_check)
        path = a.out / f"{task}_seed{seed}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["after", "eval", "regret"])
            for after, curve in out.items():
                for j, v in enumerate(curve):
                    w.writerow([after, after + j + 1, f"{v:.9g}"])
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
