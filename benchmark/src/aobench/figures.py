"""Publication figures from benchmark results.

  python -m aobench.figures --runs benchmark/runs/main [more run folders] --out benchmark/runs/figures

- regret.pdf: per task, the mean over seeds of log10 best-so-far regret against
  evaluations, with a band of one standard error.
- rank.pdf: each method's average rank over tasks (1 = best), with its rank on
  each task drawn behind it. A task's ranks come from the mean final log10 regret
  over seeds; ties share the average rank. A method missing from a task is left
  out of that task's ranking and of its own average.

Run folders are merged, so an agent run (`<out>/<task>/agent/`) can be added to
the baselines. `--seeds N` keeps seeds 0..N-1 only, so methods run on different
numbers of seeds are compared on the same instances. Scores are floored at 1e-8,
as in the report.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from .run import FLOOR  # noqa: E402
from .tasks import SPECS  # noqa: E402

# Display order, names and styles. Okabe-Ito colours (colour-blind safe); the two
# selectors are dashed and the agent is drawn heavier.
METHODS = {
    "sobol": ("Sobol", "#999999", "-"),
    "lbfgsb": ("L-BFGS-B", "#E69F00", "-"),
    "cobyla": ("COBYLA", "#56B4E9", "-"),
    "cma_es": ("CMA-ES", "#009E73", "-"),
    "smac": ("SMAC", "#CC79A7", "-"),
    "gpbo": ("GP-BO", "#D55E00", "-"),
    "shiwa": ("Shiwa", "#0072B2", "--"),
    "bandit": ("Bandit", "#000000", "--"),
    "agent": ("Agent", "#7B2D8E", "-"),
}
TITLES = {
    "ill_conditioned": "Ill-conditioned",
    "ill_conditioned_noisy": "Ill-conditioned, 1% noise",
    "noise_dominated": "Noise-dominated",
    "plateaus": "Plateaus",
    "few_evals_10d": "Few evaluations",
    "unstructured_2d": "Unstructured",
}
STYLE = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
}
WIDTH = 6.75  # inches: a two-column page


def load(runs: list[Path], seeds: int | None = None) -> dict[str, dict[str, np.ndarray]]:
    """task -> method -> (seeds, budget) array of log10 best-so-far regret."""
    data: dict[str, dict[str, np.ndarray]] = {}
    for task in SPECS:
        for method in METHODS:
            curves = {}
            for run in runs:
                for f in sorted((run / task / method).glob("seed_*.csv")):
                    if seeds is not None and int(f.stem.split("_")[1]) >= seeds:
                        continue
                    with open(f, newline="") as fh:
                        curves[f.stem] = [float(r["regret"]) for r in csv.DictReader(fh)]
            if curves:
                c = np.array(list(curves.values()))
                data.setdefault(task, {})[method] = np.log10(np.maximum(c, FLOOR))
    return data


def regret_figure(data, out: Path) -> None:
    tasks = [t for t in SPECS if t in data]
    cols = 3
    rows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(WIDTH, 2.0 * rows + 0.35), squeeze=False)
    for ax, task in zip(axes.flat, tasks):
        spec = SPECS[task]
        x = np.arange(1, spec.budget + 1)
        for method, c in data[task].items():
            name, colour, ls = METHODS[method]
            mean = c.mean(0)
            se = c.std(0, ddof=1) / np.sqrt(len(c)) if len(c) > 1 else np.zeros_like(mean)
            lw = 1.6 if method == "agent" else 1.1
            ax.plot(x, mean, color=colour, ls=ls, lw=lw, label=name, zorder=3 if method == "agent" else 2)
            ax.fill_between(x, mean - se, mean + se, color=colour, alpha=0.15, lw=0, zorder=1)
        ax.set_title(f"{TITLES.get(task, task)} ($d={spec.dim}$)")
        ax.set_xlim(1, spec.budget)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10}$ regret")
    for ax in axes[-1, :]:
        ax.set_xlabel("Evaluations")
    present = [m for m in METHODS if any(m in d for d in data.values())]
    handles = {h.get_label(): h for ax in axes.flat for h in ax.get_lines()}
    fig.legend([handles[METHODS[m][0]] for m in present], [METHODS[m][0] for m in present],
               loc="lower center", ncol=len(present), frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.0, w_pad=0.8)
    fig.align_ylabels(axes[:, 0])
    for ext in ("pdf", "png"):
        fig.savefig(out / f"regret.{ext}")
    plt.close(fig)


def ranks(data) -> dict[str, dict[str, float]]:
    """method -> task -> rank of its mean final log10 regret among the methods run on that task."""
    out: dict[str, dict[str, float]] = {}
    for task, methods in data.items():
        names = list(methods)
        final = [methods[m][:, -1].mean() for m in names]
        for m, r in zip(names, rankdata(final)):
            out.setdefault(m, {})[task] = float(r)
    return out


def rank_figure(data, out: Path) -> dict[str, dict[str, float]]:
    rk = ranks(data)
    tasks = [t for t in SPECS if t in data]
    order = sorted(rk, key=lambda m: np.mean(list(rk[m].values())))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        for j, t in enumerate(tasks):
            if t in rk[m]:
                ax.scatter(rk[m][t], i + rng.uniform(-0.18, 0.18), s=12, marker=markers[j % len(markers)],
                           facecolor="none", edgecolor="0.6", lw=0.6, zorder=1)
        avg = np.mean(list(rk[m].values()))
        ax.scatter(avg, i, s=34, color=METHODS[m][1], edgecolor="black", lw=0.5, zorder=3)
        ax.text(avg, i - 0.32, f"{avg:.1f}", ha="center", va="bottom", fontsize=6.5, zorder=4)
    labels = [METHODS[m][0] + ("$^\\dagger$" if len(rk[m]) < len(tasks) else "") for m in order]
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    n = max(len(v) for v in data.values())
    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("Rank (1 = best)")
    ax.grid(axis="x", lw=0.3, alpha=0.5)
    task_handles = [plt.Line2D([], [], ls="", marker=markers[j % len(markers)], mfc="none", mec="0.5", ms=4,
                               mew=0.6, label=TITLES.get(t, t)) for j, t in enumerate(tasks)]
    ax.legend(handles=task_handles, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False,
              fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    if any(len(rk[m]) < len(tasks) for m in order):
        fig.text(0.99, 0.07, r"${}^\dagger$ not run on every task", ha="right", va="top", fontsize=6,
                 color="0.35")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"rank.{ext}")
    plt.close(fig)
    return rk


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m aobench.figures", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", default=[Path("benchmark/runs/main")],
                   help="run folders, merged (e.g. the baselines and an agent run)")
    p.add_argument("--out", type=Path, default=Path("benchmark/runs/figures"))
    p.add_argument("--seeds", type=int, help="use seeds 0..N-1 only (default: every seed found)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    data = load(a.runs, a.seeds)
    regret_figure(data, a.out)
    rk = rank_figure(data, a.out)
    tasks = [t for t in SPECS if t in data]
    print("seeds per task and method:")
    for t in tasks:
        print(f"  {t}: " + ", ".join(f"{m} {len(c)}" for m, c in data[t].items()))
    print("rank per task (mean final log10 regret):")
    for m in sorted(rk, key=lambda m: np.mean(list(rk[m].values()))):
        cells = " ".join(f"{rk[m].get(t, float('nan')):4.1f}" for t in tasks)
        print(f"  {METHODS[m][0]:9s} avg {np.mean(list(rk[m].values())):.2f} | {cells}")
    print(f"wrote {a.out}/regret.pdf, rank.pdf (and .png)")


if __name__ == "__main__":
    main()
