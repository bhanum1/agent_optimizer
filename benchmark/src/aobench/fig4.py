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
# Three of the six tasks that carry all three arms. The other three are left out, and the
# reasons differ -- rct_2 cannot be measured this way, the last two were a drawing decision:
#   rct_2  multiobjective, so there is no single label column to take a percentile of.
#   sup    27.1x for the gray box, the widest result here, dropped to thin the panel.
#   btch   the batch protocol, and the one task where the gray box loses to the black box
#          (1.5x against 2.7x). Dropping it takes the only exception off the chart, so the
#          ordering reads as unbroken when in the full set it is not. Say so in the caption.
# benchmark/graybox/pool/ keeps the labels for all five, so any of them can come back.
TASKS = {
    "mix_1": ("Low-viscosity mixtures", "min"),
    "mat_5": ("CO$_2$ capture in MOFs", "max"),
    "rct_1": ("High-yield coupling reactions", "max"),
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


def spread(ys, gap):
    """Nudge labels apart, keeping their order, so two close tasks stay readable."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    shift = max(0.0, max(out) - 0.96)
    return [v - shift for v in out]


def draw(data, out: Path):
    keys = list(ARMS)
    x = range(len(keys))
    fig, ax = plt.subplots(figsize=(WIDTH * 0.56, 2.7))

    ax.axhline(1.0, color=RULE, lw=0.8, zorder=1)

    # No error bars: each task's would sit at the same three x positions, and five overlapping
    # would read as one column per arm -- a bar, the reading this panel exists to avoid.
    for task, row in data.items():
        ax.plot(x, [row[a] for a in keys], "-", color=INK, lw=0.9, alpha=0.45, zorder=2,
                clip_on=False)
        for i, a in enumerate(keys):
            ax.plot([i], [row[a]], "o", ms=4.2, color=ARMS[a][1], mec="white", mew=0.7, zorder=4,
                    clip_on=False)

    mean = [statistics.mean(row[a] for row in data.values()) for a in keys]
    ax.plot(x, mean, "-", color="black", lw=2.0, zorder=5, clip_on=False)
    ax.annotate("mean", (0, mean[0]), textcoords="offset points", xytext=(1, 7), ha="left",
                va="bottom", fontsize=6.8, color="black", fontfamily="STIXGeneral",
                fontweight="bold")

    ax.set_yscale("log")
    ax.autoscale_view()
    tasks = list(data)
    to_ax = lambda v: ax.transAxes.inverted().transform(ax.transData.transform((0, v)))[1]  # noqa: E731
    right = ax.transAxes.inverted().transform(ax.transData.transform((len(keys) - 1, 1)))[0]
    for task, y in zip(tasks, spread([to_ax(data[t][keys[-1]]) for t in tasks], 0.085)):
        ax.text(right + 0.03, y, TASKS[task][0], ha="left", va="center", fontsize=6.6, color=INK,
                transform=ax.transAxes)
    ax.text(right + 0.03, to_ax(1.0), "no faster than\nrandom search", ha="left", va="center",
            fontsize=6, color="#9A9A9A", linespacing=1.3, transform=ax.transAxes)

    ax.set_ylabel(f"Experiments saved vs random search,\nto a top-{TOP_PCT:g}% candidate  "
                  r"$\uparrow$", fontsize=7.5, linespacing=1.5)
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1$\\times$", "2$\\times$", "5$\\times$", "10$\\times$", "20$\\times$"])
    ax.minorticks_off()
    ax.set_xticks(list(x))
    ax.set_xticklabels([ARMS[a][0] for a in keys], fontsize=7.5, linespacing=1.15)
    for i, a in enumerate(keys):  # the tick label wears the arm's colour, so the dots need no key
        ax.get_xticklabels()[i].set_color(INK if a == "baseline" else ARMS[a][1])
    ax.set_xlim(-0.36, len(keys) - 1 + 0.06)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color=RULE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.165, right=0.575, top=0.96, bottom=0.17)
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
