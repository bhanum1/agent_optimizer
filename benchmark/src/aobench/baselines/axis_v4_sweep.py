"""Run the axis study v4 baseline sweep in parallel (see benchmark/AXIS_V4_PLAN.md).

Every v4 task x optimizer x seed, with a 300*d budget, random-start restarts and
sparse traces (checkpoints at 10, 30, 100 and 300 * d). A finished campaign has
its meta.json, so an interrupted sweep resumes where it stopped. Jobs run
largest first, each in a worker process that is replaced every few jobs, which
cleans up the solver threads that runs leave behind.

    python -m aobench.baselines.axis_v4_sweep --outdir RUNS/axis_v4 --workers 192
    python -m aobench.baselines.axis_v4_sweep --outdir RUNS/axis_v4 --list   # count only
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

from ..tasks.axis_v4 import BUDGET_PER_DIM, CHECKPOINTS_PER_DIM, axis_v4_cells

OPTIMIZERS = ["sobol", "lbfgsb", "nelder_mead", "cobyla", "cma_es"]
SEEDS = range(15)  # BBOB instances 1..15
# Rough cost per evaluation (ms, M1 Pro), only used to start the slowest jobs first.
COST = {"cobyla": 1.5}


def jobs(seeds=SEEDS, optimizers=OPTIMIZERS, keys=None):
    out = []
    for c in axis_v4_cells():
        if keys and c["key"] not in keys:
            continue
        for opt in optimizers:
            for seed in seeds:
                out.append((c["key"], c["dim"], opt, seed))
    out.sort(key=lambda j: -j[1] ** 1.3 * COST.get(j[2], 0.3))
    return out


def _run(job_and_outdir):
    (key, dim, opt, seed), outdir = job_and_outdir
    from .campaign import run_campaign
    path = Path(outdir) / key / opt / f"seed_{seed}"
    if (path / "meta.json").exists():
        return key, opt, seed, "skip", 0.0
    t = time.time()
    try:
        run_campaign(key, opt, seed, BUDGET_PER_DIM * dim, outdir, restart=True,
                     checkpoints=[k * dim for k in CHECKPOINTS_PER_DIM])
        return key, opt, seed, "ok", time.time() - t
    except Exception:
        return key, opt, seed, "error: " + traceback.format_exc(limit=3), time.time() - t


def main():
    p = argparse.ArgumentParser(prog="python -m aobench.baselines.axis_v4_sweep")
    p.add_argument("--outdir", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seeds", type=int, default=len(SEEDS))
    p.add_argument("--keys", nargs="*", help="only these task keys")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    todo = jobs(range(args.seeds), keys=set(args.keys or ()))
    print(f"{len(todo)} campaigns", flush=True)
    if args.list:
        return
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_config.json").write_text(json.dumps(
        {"optimizers": OPTIMIZERS, "seeds": args.seeds, "budget_per_dim": BUDGET_PER_DIM,
         "checkpoints_per_dim": CHECKPOINTS_PER_DIM, "restart": True, "n_campaigns": len(todo)}, indent=2))
    done = errors = 0
    t0 = time.time()
    with open(out / "sweep_log.txt", "a") as log, \
            get_context("spawn").Pool(args.workers, maxtasksperchild=25) as pool:
        for key, opt, seed, status, secs in pool.imap_unordered(_run, [(j, str(out)) for j in todo]):
            done += 1
            if status.startswith("error"):
                errors += 1
            log.write(f"{key}\t{opt}\t{seed}\t{secs:.1f}\t{status}\n")
            if done % 200 == 0 or done == len(todo):
                log.flush()
                print(f"{done}/{len(todo)} done, {errors} errors, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
