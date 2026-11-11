"""Median and quartile best-so-far log10 regret on a log-spaced evaluation grid, per v5 cell and solver.

    python -m aobench.baselines.axis_v5_curves RUNS/axis_v5      # writes RUNS/axis_v5/curves.csv

Adapted from benchmark/runs/tools/curves_v4.py. Description only: the budget is fixed at 300*d.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from ..tasks.axis_v5 import axis_v5_cells


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "curves.csv"
    cells = {c["key"]: c for c in axis_v5_cells()}
    grid_per_dim = np.unique(np.round(np.logspace(0, np.log10(300), 40), 3))
    acc = {}
    for mp in root.glob("v5_*/*/seed_*/meta.json"):
        m = json.loads(mp.read_text())
        d, fopt = m["task_dim"], m["task_f_opt"]
        with open(mp.with_name("trace.csv")) as f:
            tr = [(int(r["eval_idx"]), float(r["y_best_so_far"])) for r in csv.DictReader(f)]
        idx = np.array([i for i, _ in tr]); best = np.array([b for _, b in tr])
        evals = np.maximum(1, np.round(grid_per_dim * d)).astype(int)
        pos = np.searchsorted(idx, evals, side="right") - 1
        lr = np.where(pos >= 0, np.log10(np.maximum(best[np.maximum(pos, 0)] - fopt, 1e-8)), np.nan)
        acc.setdefault((m["task_name"], d, m["optimizer_name"]), []).append(lr)
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["key", "group", "fname", "dim", "beta", "opt", "n_seeds", "evals_per_dim", "q25", "median", "q75"])
        for (key, d, opt), runs in sorted(acc.items()):
            a = np.array(runs)
            q = np.nanpercentile(a, [25, 50, 75], axis=0)
            c = cells[key]
            for j, g in enumerate(grid_per_dim):
                w.writerow([key, c["group"], c["fname"], d, c["beta"], opt, len(runs), g, *np.round(q[:, j], 4)])
    print(len(acc), "curves")


if __name__ == "__main__":
    main()
