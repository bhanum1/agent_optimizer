"""Run every (task, solver, seed) campaign and report who is good and who is bad where.

One campaign drives one optimizer one point at a time (ask -> evaluate -> tell)
for `300 * dim` evaluations. L-BFGS-B, COBYLA and CMA-ES restart from a uniform
random point whenever they stop, so every solver uses the whole budget.

Each campaign writes `<out>/<task>/<solver>/seed_<seed>.csv` with one row per
evaluation: `eval, y, regret`. `y` is what the solver saw; `regret` is the
noise-free regret of the best point it has *observed* so far, i.e. what
"return the best point you saw" would deliver. Finished campaigns are skipped,
so an interrupted run resumes.

Score = log10 of the final regret, floored at 1e-8. A solver is **bad** on a
task when the median over seeds of score(Sobol) - score(solver) is below 0.1
decades: it does no better than quasi-random sampling.
"""

from __future__ import annotations

import csv
import os
import time
import traceback
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .baselines import OptimizerExhausted
from .tasks import BUDGET_PER_DIM, SPECS, make_task

SOLVERS = ["sobol", "lbfgsb", "cobyla", "cma_es", "turbo", "smac"]
RESTART = {"lbfgsb", "cobyla", "cma_es"}
FLOOR = 1e-8
FAIL = 0.1  # decades of advantage over Sobol


def make_solver(name: str, bounds, seed: int, budget: int):
    # Imported lazily: torch (TuRBO) and SMAC are slow to import.
    if name == "sobol":
        from .baselines.sobol import SobolSearch
        return SobolSearch(bounds, seed=seed)
    if name == "turbo":
        from .baselines.turbo import TuRBO
        return TuRBO(bounds, seed=seed)
    if name == "smac":
        from .baselines.smac import SMAC
        return SMAC(bounds, seed=seed, budget=budget)
    from .baselines.restarts import Restarting
    if name == "lbfgsb":
        from .baselines.lbfgsb import LBFGSB
        return Restarting(LBFGSB, bounds, seed)
    if name == "cobyla":
        from .baselines.cobyla import COBYLA
        return Restarting(COBYLA, bounds, seed)
    if name == "cma_es":
        from .baselines.cma_es import CMAES
        return Restarting(CMAES, bounds, seed, stop_when_converged=True)
    raise ValueError(f"unknown solver {name!r}; choose from {SOLVERS}")


def run_campaign(task_key: str, solver: str, seed: int, dim: int, out: Path) -> Path:
    path = out / task_key / solver / f"seed_{seed}.csv"
    if path.exists():
        return path
    warnings.filterwarnings("ignore", message=".*balance properties.*")  # scipy Sobol, one point at a time
    warnings.filterwarnings("ignore", message=".*not p.d., added jitter.*")  # GPyTorch, TuRBO
    task = make_task(task_key, dim=dim, seed=seed)
    budget = BUDGET_PER_DIM * dim
    opt = make_solver(solver, task.bounds, seed, budget)
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
    return path


def _job(args):
    t = time.time()
    try:
        run_campaign(*args)
        return args, None, time.time() - t
    except Exception:
        return args, traceback.format_exc(limit=3), time.time() - t


def run_all(out: Path, tasks, solvers, seeds, dim: int, workers: int) -> None:
    jobs = [(t, s, seed, dim, out) for t in tasks for s in solvers for seed in seeds
            if not (out / t / s / f"seed_{seed}.csv").exists()]
    jobs.sort(key=lambda j: j[1] not in ("turbo", "smac"))  # slow solvers first
    print(f"{len(jobs)} campaigns to run", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with get_context("spawn").Pool(min(workers, len(jobs)), maxtasksperchild=10) as pool:
        for k, (args, err, secs) in enumerate(pool.imap_unordered(_job, jobs), 1):
            tag = f"{args[0]} {args[1]} seed {args[2]}"
            if err:
                print(f"ERROR {tag}:\n{err}", flush=True)
            elif args[1] in ("turbo", "smac") or k % 50 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}, {time.time() - t0:.0f}s] {tag}: {secs:.0f}s", flush=True)


def scores(out: Path, task: str, solver: str) -> list[float]:
    vals = []
    for p in sorted((out / task / solver).glob("seed_*.csv")):
        with open(p) as f:
            last = list(csv.reader(f))[-1]
        vals.append(float(np.log10(max(float(last[2]), FLOOR))))
    return vals


def report(out: Path, tasks, solvers) -> str:
    """Markdown: median final log10 regret per task and solver, and the good/bad verdict."""
    lines = ["Median final log10 regret over seeds (lower is better). "
             f"**bold** = best on the task; ✗ = bad (less than {FAIL} decades ahead of Sobol).", "",
             "| task | " + " | ".join(solvers) + " | expected good | expected bad |",
             "|---" * (len(solvers) + 3) + "|"]
    for t in tasks:
        med = {s: float(np.median(v)) if (v := scores(out, t, s)) else None for s in solvers}
        sob = scores(out, t, "sobol")
        done = [m for m in med.values() if m is not None]
        cells = []
        for s in solvers:
            if med[s] is None:
                cells.append("–")
                continue
            c = f"{med[s]:.2f}"
            if s != "sobol" and sob:
                v = scores(out, t, s)
                n = min(len(v), len(sob))
                if np.median(np.array(sob[:n]) - np.array(v[:n])) < FAIL:
                    c += " ✗"
            if med[s] == min(done):
                c = f"**{c}**"
            cells.append(c)
        _, _, good, bad = SPECS[t]
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {good} | {bad} |")
    return "\n".join(lines) + "\n"


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
