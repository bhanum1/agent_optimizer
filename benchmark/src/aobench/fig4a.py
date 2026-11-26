"""Figure 4a: the three information arms, task by task.

The measure is the acceleration factor used in the materials-discovery literature (Rohr et
al.; Liang et al., npj Comput. Mater. 2021): how many times fewer experiments a method needs
than random search to turn up a top-1% candidate. It reads without a definition -- 24x means
random search needed 24 times as many experiments -- and the top-k% target is what keeps it
finite, since random search almost never finds the single best row of a pool this size.

The target has to be a demanding one. At top 5% random search already arrives in 11 to 34
experiments on most of these pools, so every method's first evaluations -- which are a Sobol
or random design -- get there by luck and the factors collapse toward 1x. Top 1% leaves room
to be fast in.

One group per task, three bars in it, so the ordering (baselines, then the agent without the
variable descriptions, then the agent with them) is read six times over rather than once in an
average, and the tasks that break it are visible as themselves. The bars grow from the 1x line
because 1x is the real zero of a ratio: below it a method is slower than random search. The
axis is logarithmic for the same reason -- 2x and 1/2x are the same size of effect.

The pool labels are needed only to find each task's top-1% threshold -- offline scoring, never
anything an agent sees -- so only the label column is kept, in benchmark/graybox/pool/.

    .venv/bin/python -m aobench.fig4 --out benchmark/figures/fig4/fig4a
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .figures import STYLE, WIDTH  # noqa: E402

TOP_PCT = 1.0
# The six tasks of the gray-box campaign, in its own order. mat_1 / mat_2 / mat_6 also carry
# all three arms and could be added, but mat_5 is already the MOF question and they are it
# again over other pools and budgets.
TASKS = {
    # rct_2 is the sixth task of the campaign and is left out here: it is multiobjective, so
    # there is no single column to take a percentile of.
    "mix_1": ("Low-viscosity\nmixtures", "min"),
    "sup": ("High-temp.\nsuperconductors", "max"),
    "mat_5": ("CO$_2$ capture\nin MOFs", "max"),
    "rct_1": ("High-yield\ncoupling", "max"),
    "btch": ("Methane\ncoupling", "max"),
}
REFERENCE = ["pool_random", "batch_random"]  # what the acceleration is measured against
ARMS = {
    # The baseline bar is the best of whatever classical optimizers the task carries, so the
    # agent is never credited for beating a method nobody would have chosen. It flatters the
    # baselines: the best one per task is picked after the fact, which is an oracle a
    # practitioner does not have. gp_qlogehvi is the multiobjective one, batch_gp_qlogei the
    # batch one; each is the only classical method its protocol admits.
    "baseline": ("Best baseline", "#9AA3AD",
                 ["gp_bo", "sobol_gp_bo", "turbo", "batch_gp_qlogei", "gp_qlogehvi"]),
    "blackbox": ("Agent, black box", "#D55E00", ["cheapest_lunch_opus_5_5_high"]),
    "graybox": ("Agent, gray box", "#7B2D8E", ["cheapest_lunch_opus_5_5_high_domain_context"]),
}
INK = "#3C3C3C"
RULE = "#D9D9D9"


def target(pool: Path, task: str):
    """`reached(value)`: is this measurement in the pool's best TOP_PCT percent?"""
    sense = TASKS[task][1]
    ys = np.loadtxt(pool / f"{task}.csv", skiprows=1)
    thr = float(np.percentile(ys, TOP_PCT if sense == "min" else 100 - TOP_PCT))
    return (lambda v: v <= thr) if sense == "min" else (lambda v: v >= thr)


def runs(results: Path, task: str, methods: list[str]) -> dict[str, list[list[tuple[float, float]]]]:
    """method -> one (evaluation, measured value) curve per seed."""
    col = "raw_value"
    out: dict[str, list[list[tuple[float, float]]]] = {}
    for m in methods:
        d = results / "tasks" / task / m
        for seed in sorted(d.glob("seed_*")) if d.is_dir() else []:
            f = seed / "trace.csv"
            if not f.exists():
                continue
            rows = list(csv.DictReader(open(f)))
            if not rows or col not in rows[0]:
                continue
            ev = "eval_idx" if "eval_idx" in rows[0] else "batch_completed_evaluations"
            c = [(float(r[ev]), float(r[col])) for r in rows if r.get(col)]
            if c:
                out.setdefault(m, []).append(c)
    return out


def collect(results: Path, pool: Path):
    """task -> arm -> acceleration factor over random search, and the raw evaluation counts."""
    data, raw = {}, {}
    for task in TASKS:
        reached = target(pool, task)
        groups = {a: runs(results, task, ms) for a, (_, _, ms) in ARMS.items()}
        groups["random"] = runs(results, task, REFERENCE)
        if not all(groups.values()):
            continue
        budget = max(e for g in groups.values() for cs in g.values() for c in cs for e, _ in c)

        def evals(c):  # a run that never gets there is capped at the budget, so the factor
            return next((e for e, v in c if reached(v)), budget)  # understates rather than breaks

        row = {}
        for arm, by_method in groups.items():
            per_method = {m: statistics.mean(evals(c) for c in cs) for m, cs in by_method.items()}
            # the baseline arm is the best classical optimizer on this task; the others are one
            # method each, so the min is just that method.
            row[arm] = min(per_method.values())
        raw[task] = row
        data[task] = {a: row["random"] / row[a] for a in ARMS}
    return data, raw


def geomean(xs):
    """The right average for a ratio: 2x then 8x averages to 4x, not 5x."""
    return float(np.exp(np.mean(np.log(list(xs)))))


def bars(ax, data):
    """The panel itself, on an axes someone else owns: fig4.py stacks it over b and c."""
    keys = list(ARMS)
    tasks = list(data)

    w = 0.26  # three of these span 0.78 of the 1.0 slot: a clear gap between groups
    lo = min(v for row in data.values() for v in row.values())
    hi = max(v for row in data.values() for v in row.values())
    for i, a in enumerate(keys):
        xs = [t + (i - 1) * w for t in range(len(tasks))]
        vs = [data[t][a] for t in tasks]
        # Bars grow from 1x, the real zero of a ratio: a bar below the line is slower than
        # random search. Drawn as a height from bottom=1 so the sign shows as direction.
        ax.bar(xs, [v - 1 for v in vs], bottom=1, width=w * 0.92, color=ARMS[a][1],
               label=ARMS[a][0], zorder=2, linewidth=0)

    ax.set_yscale("log")
    ax.axhline(1.0, color="#8A8A8A", lw=0.8, zorder=3)
    ax.set_ylim(min(lo, 1) * 0.80, hi * 1.30)
    ticks = [1, 2, 5, 10, 20]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v}$\\times$" for v in ticks])
    ax.minorticks_off()
    ax.set_ylabel(f"Acceleration factor to a top-{TOP_PCT:g}% candidate  " r"$\uparrow$",
                  fontsize=8)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASKS[t][0] for t in tasks], fontsize=6.6, linespacing=1.3, color=INK)
    ax.set_xlim(-0.6, len(tasks) - 0.4)
    ax.tick_params(axis="x", length=0, pad=4)

    # Each arm's average as a rule across the whole panel: the three levels are then a thing
    # you see at once, and every bar is read against its own arm's line as well as its
    # neighbours'. Dashed, so no rule is mistaken for a bar's edge.
    means = {a: geomean(row[a] for row in data.values()) for a in keys}
    for a in keys:
        ax.axhline(means[a], color=ARMS[a][1], lw=1.0, ls=(0, (4, 2.5)), zorder=5, alpha=0.9)
        ax.text(len(tasks) - 0.40, means[a], f" {means[a]:.1f}$\\times$", ha="left", va="center",
                fontsize=7, color=ARMS[a][1], fontfamily="STIXGeneral", fontweight="bold",
                clip_on=False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False,
              fontsize=7.5, handlelength=1.0, handleheight=0.9, columnspacing=1.6,
              handletextpad=0.5)

    ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# The panel's own margins, as fractions of the standalone figure below; fig4.py reads them
# so the stacked panel keeps the proportions this one was laid out with.
MARGINS = dict(left=0.105, right=0.935, top=0.88, bottom=0.17)
SIZE = (WIDTH * 0.78, 2.75)


def draw(data, out: Path):
    fig, ax = plt.subplots(figsize=SIZE)
    bars(ax, data)
    fig.subplots_adjust(**MARGINS)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=400)
    print(f"wrote {out} (.png and .pdf)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=Path("benchmark/graybox"))
    p.add_argument("--pool", type=Path, default=Path("benchmark/graybox/pool"),
                   help="one label column per task, for the top-1%% cut")
    p.add_argument("--out", type=Path, default=Path("benchmark/figures/fig4/fig4a"))
    p.add_argument("--top", type=float, default=TOP_PCT, help="percentile that counts as a hit")
    a = p.parse_args(argv)
    globals()["TOP_PCT"] = a.top
    data, raw = collect(a.results, a.pool)
    if not data:
        raise SystemExit(f"no task in {a.results} has all three arms and a reference")
    with plt.rc_context(STYLE):
        draw(data, a.out)
    print(f"  {'task':8} {'random':>8} {'base':>8} {'black':>8} {'gray':>8}   acceleration")
    for t, row in raw.items():
        print(f"  {t:8} " + "".join(f"{row[a]:8.0f}" for a in ("random", *ARMS)) + "   "
              + "  ".join(f"{data[t][a]:.1f}x" for a in ARMS))
    print("  geomean  " + " " * 8
          + "".join(f"{geomean(data[t][a] for t in data):8.1f}" for a in ARMS))


if __name__ == "__main__":
    main()
