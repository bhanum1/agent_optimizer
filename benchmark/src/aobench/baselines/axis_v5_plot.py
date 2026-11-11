"""Regret-vs-evaluations curves for axis study v5: one figure per group, a panel per function x d.

    python -m aobench.baselines.axis_v5_plot RUNS/axis_v5      # reads curves.csv, writes plots/<group>.png

Adapted from benchmark/runs/tools/plot_curves.py; adds TuRBO and the v5 group names.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..tasks.axis_v5 import GROUPS

INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
STYLE = {"sobol": ("Sobol", "#8a8980", (0, (4, 2))), "lbfgsb": ("L-BFGS-B", "#2a78d6", "-"),
         "nelder_mead": ("Nelder–Mead", "#eb6834", (0, (1, 1.5))), "cobyla": ("COBYLA", "#1baf7a", (0, (6, 2, 1, 2))),
         "cma_es": ("CMA-ES", "#eda100", "-"), "turbo": ("TuRBO", "#9b3fc2", (0, (3, 1, 1, 1, 1, 1)))}
TITLE = {"plateau": "Plateau", "noise_moderate": "Moderate noise, β = 0.01", "noise_severe": "Severe noise, β = 1",
         "katsuura": "Katsuura", "weierstrass": "Weierstrass", "illcond": "Ill-conditioned"}
FTITLE = {"ellipsoid": "f10 ellipsoid", "discus": "f11 discus", "bentcigar": "f12 bent cigar", "sphere": "f1 sphere",
          "rosenbrockrot": "f9 Rosenbrock", "weierstrass": "f16 Weierstrass", "katsuura": "f23 Katsuura",
          "stepellipsoid": "f7 step ellipsoid"}


def main():
    root = Path(sys.argv[1]); out = root / "plots"; out.mkdir(exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    c = pd.read_csv(root / "curves.csv")
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF})
    for group, spec in GROUPS.items():
        g = c[c.group == group]
        if g.empty:
            continue
        fnames = [fn for _, fn in spec["functions"]]
        dims = sorted(g.dim.unique())
        fig, axes = plt.subplots(len(fnames), len(dims), figsize=(3.4 * len(dims) + 0.4, 2.5 * len(fnames) + 0.9),
                                 squeeze=False)
        for i, fn in enumerate(fnames):
            for j, d in enumerate(dims):
                ax = axes[i, j]
                cell = g[(g.dim == d) & (g.fname == fn)]
                for opt, (lab, col, ls) in STYLE.items():
                    s = cell[cell.opt == opt]
                    if s.empty:
                        continue
                    ax.fill_between(s.evals_per_dim * d, s.q25, s.q75, color=col, alpha=0.12, lw=0)
                    ax.plot(s.evals_per_dim * d, s["median"], color=col, ls=ls, lw=2, label=lab)
                ax.set_xscale("log")
                ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
                for sp in ["top", "right"]:
                    ax.spines[sp].set_visible(False)
                ax.tick_params(length=0)
                ax.set_title(f"{FTITLE[fn]}, d = {d}", fontsize=9, loc="left", color=INK)
                if j == 0:
                    ax.set_ylabel("log₁₀ regret (median)")
                if i == len(fnames) - 1:
                    ax.set_xlabel("evaluations")
        fig.suptitle(f"{TITLE[group]} (median, IQR band; 15 seeds)", x=0.01, ha="left", fontsize=12,
                     fontweight="bold", color=INK)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", ncol=6, frameon=False, fontsize=9, handlelength=3)
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.75 / fig.get_size_inches()[1]])
        fn_out = out / f"{group}.png"
        fig.savefig(fn_out, dpi=140); plt.close(fig); print(fn_out)


if __name__ == "__main__":
    main()
