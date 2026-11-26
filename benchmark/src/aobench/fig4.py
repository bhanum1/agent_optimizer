"""Figure 4a: the three information arms on the real tasks, as an ordering.

The measure is the acceleration factor used in the materials-discovery literature (Rohr et
al.; Liang et al., npj Comput. Mater. 2021): how many times fewer experiments a method needs
than random search to turn up a top-1% candidate. It reads without a definition -- 24x means
random search needed 24 times as many experiments -- and the top-k% target is what keeps it
finite, since random search almost never finds the single best row of a pool this size.

The panel's point is not the distance between any two arms on any one task. It is that the
order holds: baselines, then the agent without the variable descriptions, then the agent with
them. Bars would invite the reader to measure heights against each other; a slope between
three positions puts the ordering itself on the page, and a task that breaks it shows as a
crossing line rather than disappearing into an average.

Colour carries nothing on the lines: they are one ink and the tasks are named where they end,
so no two tasks are set against each other by hue. The three arms are coloured, because they
are the thing being ordered.

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
# Every task that carries all three arms. rct_2 is the one exclusion: it is multiobjective,
# so it has no single label column to take a percentile of.
TASKS = {
    "mix_1": ("Low-viscosity mixtures", "min"),
    "sup": ("High-temp. superconductors", "max"),
    "mat_5": ("CO$_2$ capture in MOFs", "max"),
    "rct_1": ("High-yield coupling reactions", "max"),
    "btch": ("Methane coupling catalysts", "max"),
}
REFERENCE = ["pool_random", "batch_random"]  # what the acceleration is measured against
ARMS = {
    "baseline": ("Baselines", "#9AA3AD", ["gp_bo", "sobol_gp_bo", "turbo", "batch_gp_qlogei"]),
    "blackbox": ("Agent\nblack box", "#D55E00", ["cheapest_lunch_opus_5_5_high"]),
    "graybox": ("Agent\ngray box", "#7B2D8E", ["cheapest_lunch_opus_5_5_high_domain_context"]),
}
INK = "#3C3C3C"
RULE = "#D9D9D9"


def threshold(pool: Path, task: str) -> float:
    """The value a candidate must beat to be in the task's best 1% of the pool."""
    ys = np.loadtxt(pool / f"{task}.csv", skiprows=1)
    return float(np.percentile(ys, TOP_PCT if TASKS[task][1] == "min" else 100 - TOP_PCT))


def runs(results: Path, task: str, methods: list[str]) -> dict[str, list[list[tuple[float, float]]]]:
    """method -> one (evaluation, measured value) curve per seed."""
    out: dict[str, list[list[tuple[float, float]]]] = {}
    for m in methods:
        d = results / "tasks" / task / m
        for seed in sorted(d.glob("seed_*")) if d.is_dir() else []:
            f = seed / "trace.csv"
            if not f.exists():
                continue
            rows = list(csv.DictReader(open(f)))
            if not rows or "raw_value" not in rows[0]:
                continue
            ev = "eval_idx" if "eval_idx" in rows[0] else "batch_completed_evaluations"
            c = [(float(r[ev]), float(r["raw_value"])) for r in rows if r.get("raw_value")]
            if c:
                out.setdefault(m, []).append(c)
    return out


def collect(results: Path, pool: Path):
    """task -> arm -> acceleration factor over random search, and the raw evaluation counts."""
    data, raw = {}, {}
    for task, (_, sense) in TASKS.items():
        thr = threshold(pool, task)
        reached = (lambda v: v <= thr) if sense == "min" else (lambda v: v >= thr)
        groups = {a: runs(results, task, ms) for a, (_, _, ms) in ARMS.items()}
        groups["random"] = runs(results, task, REFERENCE)
        if not all(groups.values()):
            continue
        budget = max(e for g in groups.values() for cs in g.values() for c in cs for e, _ in c)

        def evals(c):  # a run that never gets there is capped at the budget, so the factor
            return next((e for e, v in c if reached(v)), budget)  # understates rather than breaks

        row = {}
        for arm, by_method in groups.items():
            if arm == "baseline":  # the best single baseline, not an average over a mixed bag
                by_method = dict([min(by_method.items(),
                                      key=lambda kv: statistics.mean(evals(c) for c in kv[1]))])
            row[arm] = statistics.mean(evals(c) for cs in by_method.values() for c in cs)
        raw[task] = row
        data[task] = {a: row["random"] / row[a] for a in ARMS}
    return data, raw


def geomean(xs):
    """The right average for a ratio: 2x then 8x averages to 4x, not 5x."""
    return float(np.exp(np.mean(np.log(list(xs)))))


def draw(data, out: Path):
    keys = list(ARMS)
    fig, ax = plt.subplots(figsize=(WIDTH * 0.50, 2.75))

    bars = [geomean(row[a] for row in data.values()) for a in keys]
    ax.bar(range(len(keys)), bars, width=0.56, color=[ARMS[a][1] for a in keys], zorder=2)
    top = max(row[a] for row in data.values() for a in keys)

    # One dot per task on each bar, so the whole set is on the page and the spread behind each
    # average is visible. The dots are one ink: which task is which is not this panel's claim.
    rng = np.random.default_rng(0)
    for row in data.values():
        for i, a in enumerate(keys):
            ax.plot([i + rng.uniform(-0.13, 0.13)], [row[a]], "o", ms=3.4, mfc="white",
                    mec="#4A4A4A", mew=0.8, zorder=4, alpha=0.95)

    ax.axhline(1.0, color="#8A8A8A", lw=0.7, ls=(0, (3, 2)), zorder=3)
    # Each column's number goes above everything in that column, clear of its own dots.
    for i, v in enumerate(bars):
        high = max(row[keys[i]] for row in data.values())
        ax.text(i, high + top * 0.035, f"{v:.1f}$\\times$", ha="center", va="bottom",
                fontsize=9, color=ARMS[keys[i]][1], fontfamily="STIXGeneral", fontweight="bold")

    ax.set_ylabel(f"Experiments to a top-{TOP_PCT:g}% candidate,\nversus random search  "
                  r"$\uparrow$", fontsize=7.5, linespacing=1.5)
    ax.set_ylim(0, top * 1.14)
    ax.set_yticks([1, 5, 10, 15, 20, 25])  # no 0: the spine is the zero, and it crowds the 1x
    ax.set_yticklabels([f"{v}$\\times$" for v in (1, 5, 10, 15, 20, 25)])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([ARMS[a][0] for a in keys], fontsize=7.5, linespacing=1.15)
    for i, a in enumerate(keys):
        ax.get_xticklabels()[i].set_color(INK if a == "baseline" else ARMS[a][1])
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.text(-0.55, -0.225, f"Bars: geometric mean over {len(data)} tasks.  Dots: one task each."
            "\nDashed line: no faster than random search.", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=6, color="#7A7A7A", linespacing=1.6)
    ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.20, right=0.98, top=0.97, bottom=0.26)
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
    a = p.parse_args(argv)
    data, raw = collect(a.results, a.pool)
    if not data:
        raise SystemExit(f"no task in {a.results} has all three arms and a reference")
    with plt.rc_context(STYLE):
        draw(data, a.out)
    print(f"  {'task':8} {'random':>8} {'base':>8} {'black':>8} {'gray':>8}   acceleration")
    for t, row in raw.items():
        print(f"  {t:8} " + "".join(f"{row[a]:8.0f}" for a in ("random", *ARMS)) + "   "
              + "  ".join(f"{data[t][a]:.1f}x" for a in ARMS))


if __name__ == "__main__":
    main()
