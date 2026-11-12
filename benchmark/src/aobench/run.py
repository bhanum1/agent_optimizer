"""Run every (task, solver, seed) campaign and report who is good and who is bad where.

One campaign drives one optimizer one point at a time (ask -> evaluate -> tell)
for the task's budget. L-BFGS-B, COBYLA and CMA-ES restart from a uniform
random point whenever they stop, so every solver uses the whole budget.

Each campaign writes `<out>/<task>/<solver>/seed_<seed>.csv` with one row per
evaluation: `eval, y, regret`. `y` is what the solver saw; `regret` is the
noise-free regret of the best point it has *observed* so far, i.e. what
"return the best point you saw" would deliver. Finished campaigns are skipped,
so an interrupted run resumes.

Score = log10 of the final regret, floored at 1e-8. Two kinds of failure, both
on medians over seeds:
- **soft fail** (◦): the solver's median score is at least SOFT = 1 decade worse
  than the best solver's on that task;
- **hard fail** (✗): it is less than 0.1 decades ahead of Sobol, i.e. no better
  than quasi-random sampling.
"""

from __future__ import annotations

import csv
import json
import os
import time
import traceback
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .baselines import SELECTORS, SLOW, OptimizerExhausted, make_solver
from .tasks import SPECS, make_task

FLOOR = 1e-8
FAIL = 0.1  # hard fail: less than this many decades ahead of Sobol
SOFT = 1.0  # soft fail: at least this many decades behind the best solver


def run_campaign(task_key: str, solver: str, seed: int, out: Path) -> Path:
    path = out / task_key / solver / f"seed_{seed}.csv"
    if path.exists():
        return path
    warnings.filterwarnings("ignore", message=".*balance properties.*")  # scipy Sobol, one point at a time
    warnings.filterwarnings("ignore", message=".*not p.d., added jitter.*")  # GPyTorch, GP-BO
    task = make_task(task_key, seed=seed)
    budget = task.spec.budget
    opt = make_solver(solver, task.bounds, seed, budget, noisy=task.spec.beta > 0)
    rows, best_y, regret = [], np.inf, np.inf
    for i in range(budget):
        try:
            X = opt.ask(1)
        except OptimizerExhausted:
            break
        ev = task.evaluate(X[0])
        opt.tell(X, [ev.y])
        if ev.y < best_y:
            best_y, regret = ev.y, ev.y_true - task.f_opt
        rows.append((i + 1, ev.y, regret))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval", "y", "regret"])
        w.writerows(rows)
    tmp.rename(path)
    if hasattr(opt, "metadata"):  # the selectors say what they picked
        path.with_suffix(".json").write_text(json.dumps(opt.metadata(), indent=1, default=float))
    return path


def _job(args):
    t = time.time()
    try:
        run_campaign(*args)
        return args, None, time.time() - t
    except Exception:
        return args, traceback.format_exc(limit=3), time.time() - t


def run_all(out: Path, tasks, solvers, seeds, workers: int) -> None:
    jobs = [(t, s, seed, out) for t in tasks for s in solvers for seed in seeds
            if not (out / t / s / f"seed_{seed}.csv").exists()]
    jobs.sort(key=lambda j: j[1] not in SLOW)  # slow solvers first
    print(f"{len(jobs)} campaigns to run", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=10) as pool:
        for k, (args, err, secs) in enumerate(pool.imap_unordered(_job, jobs), 1):
            tag = f"{args[0]} {args[1]} seed {args[2]}"
            if err:
                print(f"ERROR {tag}:\n{err}", flush=True)
            elif args[1] in SLOW or k % 50 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}, {time.time() - t0:.0f}s] {tag}: {secs:.0f}s", flush=True)


def scores(out: Path, task: str, solver: str) -> list[float]:
    vals = []
    for p in sorted((out / task / solver).glob("seed_*.csv")):
        with open(p) as f:
            last = list(csv.reader(f))[-1]
        vals.append(float(np.log10(max(float(last[2]), FLOOR))))
    return vals


def report(out: Path, tasks, solvers) -> str:
    """Markdown: median final log10 regret per task and solver, with soft and hard failures."""
    lines = ["Median final log10 regret over seeds (lower is better). **bold** = best on the task; "
             f"◦ = soft fail (≥ {SOFT:g} decade behind the best); ✗ = hard fail (less than {FAIL} "
             f"decades ahead of Sobol). {' and '.join(sorted(SELECTORS))} are reported but never set the best.", "",
             "| task | d | budget | seeds | " + " | ".join(solvers) + " | expected good | expected bad |",
             "|---" * (len(solvers) + 6) + "|"]
    for t in tasks:
        sc = {s: scores(out, t, s) for s in solvers}
        med = {s: float(np.median(v)) for s, v in sc.items() if v}
        if not med:
            continue
        best = min(v for s, v in med.items() if s not in SELECTORS)
        sob = sc.get("sobol", [])
        cells = []
        for s in solvers:
            if s not in med:
                cells.append("–")
                continue
            c = f"{med[s]:.2f}"
            n = min(len(sc[s]), len(sob))
            if s != "sobol" and n and np.median(np.array(sob[:n]) - np.array(sc[s][:n])) < FAIL:
                c += " ✗"
            elif med[s] - best >= SOFT:
                c += " ◦"
            cells.append(f"**{c}**" if med[s] == best and s not in SELECTORS else c)
        spec = SPECS[t]
        n_seeds = max(len(v) for v in sc.values())
        lines.append(f"| {t} | {spec.dim} | {spec.budget} | {n_seeds} | " + " | ".join(cells)
                     + f" | {spec.good} | {spec.bad} |")
    return "\n".join(lines) + "\n"


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
