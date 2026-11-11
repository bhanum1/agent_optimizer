"""Run the axis study v5 baseline sweep in parallel (see benchmark/AXIS_V5_PLAN.md).

Every v5 task x optimizer x seed, with a 300*d budget, random-start restarts for
the local solvers and CMA-ES (TuRBO restarts its own trust region), and sparse
traces (checkpoints at 10, 30, 100 and 300 * d). A finished campaign has its
meta.json, so an interrupted sweep resumes where it stopped. Jobs run slowest
first, each in a worker process that is replaced every few jobs, which cleans up
the solver threads that runs leave behind.

The job list can be cut by optimizer and dimension, and split into shards
(job i of n takes every n-th job of the cost-sorted list), so TuRBO can be
spread over several nodes:

    python -m aobench.baselines.axis_v5_sweep --outdir RUNS/axis_v5 --workers 190
    python -m aobench.baselines.axis_v5_sweep --outdir RUNS/axis_v5 --optimizers turbo --shard 0/4
    python -m aobench.baselines.axis_v5_sweep --outdir RUNS/axis_v5 --list   # count only
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

from ..tasks.axis_v5 import BUDGET_PER_DIM, CHECKPOINTS_PER_DIM, DIMS, axis_v5_cells

OPTIMIZERS = ["sobol", "lbfgsb", "nelder_mead", "cobyla", "cma_es", "turbo"]
SEEDS = range(15, 30)  # BBOB instances 16..30 (v4 used 1..15)
# Rough cost per evaluation (ms), only used to start the slowest jobs first.
COST = {"cobyla": 1.5, "turbo": 1000.0}


def jobs(seeds=SEEDS, optimizers=OPTIMIZERS, dims=DIMS, keys=None):
    out = []
    for c in axis_v5_cells(dims):
        if keys and c["key"] not in keys:
            continue
        for opt in optimizers:
            for seed in seeds:
                out.append((c["key"], c["dim"], opt, seed))
    out.sort(key=lambda j: (-j[1] ** 1.3 * COST.get(j[2], 0.3), j[0], j[3]))
    return out


def _run(job_and_outdir):
    (key, dim, opt, seed), outdir = job_and_outdir
    from .campaign import run_campaign
    path = Path(outdir) / key / opt / f"seed_{seed}"
    if (path / "meta.json").exists():
        return key, opt, seed, "skip", 0.0
    t = time.time()
    try:
        # restart=True only affects campaign.RESTARTABLE solvers; TuRBO runs as-is.
        run_campaign(key, opt, seed, BUDGET_PER_DIM * dim, outdir, restart=True,
                     checkpoints=[k * dim for k in CHECKPOINTS_PER_DIM])
        return key, opt, seed, "ok", time.time() - t
    except Exception:
        return key, opt, seed, "error: " + traceback.format_exc(limit=3), time.time() - t


def main():
    p = argparse.ArgumentParser(prog="python -m aobench.baselines.axis_v5_sweep")
    p.add_argument("--outdir", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--optimizers", nargs="*", default=OPTIMIZERS, choices=OPTIMIZERS)
    p.add_argument("--dims", nargs="*", type=int, default=DIMS)
    p.add_argument("--keys", nargs="*", help="only these task keys")
    p.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    i, n = map(int, args.shard.split("/"))
    todo = jobs(optimizers=args.optimizers, dims=args.dims, keys=set(args.keys or ()))[i::n]
    print(f"{len(todo)} campaigns", flush=True)
    if args.list:
        for j in todo:
            print(*j, sep="\t")
        return
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{'-'.join(args.optimizers)}_d{'-'.join(map(str, args.dims))}_shard{i}of{n}"
    (out / f"sweep_config_{tag}.json").write_text(json.dumps(
        {"optimizers": args.optimizers, "dims": args.dims, "seeds": list(SEEDS), "shard": args.shard,
         "budget_per_dim": BUDGET_PER_DIM, "checkpoints_per_dim": CHECKPOINTS_PER_DIM,
         "restart": True, "n_campaigns": len(todo)}, indent=2))
    done = errors = 0
    t0 = time.time()
    with open(out / f"sweep_log_{tag}.txt", "a") as log, \
            get_context("spawn").Pool(min(args.workers, max(len(todo), 1)), maxtasksperchild=25) as pool:
        for key, opt, seed, status, secs in pool.imap_unordered(_run, [(j, str(out)) for j in todo]):
            done += 1
            if status.startswith("error"):
                errors += 1
            log.write(f"{key}\t{opt}\t{seed}\t{secs:.1f}\t{status}\n")
            log.flush()
            if done % 200 == 0 or done == len(todo) or opt == "turbo":
                print(f"{done}/{len(todo)} done, {errors} errors, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
