"""Figure 3 as an animation: the optimizer's walk, and what each decision does to it.

    .venv/bin/python -m aobench.fig3_anim --out benchmark/figures/fig3/fig3.gif

Left, the campaign's evaluations in the plane of two of the ten variables (the optimum is the
star); right, regret against evaluations. The walk is paused at each key decision: its
reasoning appears across the top, and the campaign it replaced -- Figure 3's counterfactual --
runs out as a dashed line, so the jump that follows can be read against what would have
happened. A projection to two variables is a view, not the whole space: the points that look
far from the star may be close in the other eight.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from .fig2 import SCALE  # noqa: E402
from .fig3 import CAMPAIGNS, COLOURS, FLOOR, load, workspace  # noqa: E402
from .figures import METHODS, STYLE  # noqa: E402

AGENT = METHODS["agent"][1]


def points(path: Path, task):
    """Each measured evaluation's x, and the label of the block that proposed it."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "measured" and r["y"] != ""]
    rows = rows[:task.spec.budget]
    return np.array([[float(r[f"x{i}"]) for i in range(task.dim)] for r in rows]), [r["label"] for r in rows]


def landscape(task, d0, d1, n=161):
    """log10 regret on the plane through the optimum spanned by two variables, as Figure 2a."""
    xo = np.asarray(task._problem.optimum.x, dtype=float)
    g = np.linspace(-5, 5, n)
    z = np.empty((n, n))
    for i, b in enumerate(g):
        for j, a in enumerate(g):
            x = xo.copy()
            x[d0], x[d1] = a, b
            z[i, j] = task._problem(x) - task.f_opt
    return np.log10(np.maximum(z, 1e-8))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--campaign", default="few_evals_10d/1", choices=sorted(CAMPAIGNS))
    p.add_argument("--runs", type=Path, default=Path("benchmark/runs/opus_high"))
    p.add_argument("--baselines", type=Path, default=Path("benchmark/runs/main"))
    p.add_argument("--branches", type=Path, default=Path("benchmark/runs/counterfactual"))
    p.add_argument("--out", type=Path, default=Path("benchmark/figures/fig3/fig3.gif"))
    p.add_argument("--dims", type=int, nargs=2, default=(0, 1))
    a = p.parse_args(argv)
    task, agent, base, decisions = load(a.campaign, a.runs, a.baselines, a.branches)
    key, seed = a.campaign.split("/")
    X, labels = points(workspace(a.runs, key, int(seed)) / ".agentoptimizer" / "data.csv", task)
    opt = np.asarray(task._problem.optimum.x)
    d0, d1 = a.dims
    n = len(agent)

    # timeline: (kind, evals shown, decision index, branch evals shown)
    STEP, HOLD, DRAW = 3, 14, 22
    frames, cuts = [], [d["after"] for d in decisions]
    e = 0
    while e < n:
        e = min(e + STEP, n)
        for i, c in enumerate(cuts):
            if e >= c and not any(f[1] == i for f in frames):
                frames += [(c, i, 0)] * HOLD
                frames += [(c, i, int((n - c) * (k + 1) / DRAW)) for k in range(DRAW)]
                frames += [(c, i, n - c)] * HOLD
        if e not in cuts or e == n:
            frames.append((e, None, None))
    frames += [(n, None, None)] * 25

    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(9.0, 5.2), dpi=110)
    axp = fig.add_axes([0.05, 0.09, 0.36, 0.62])
    axr = fig.add_axes([0.53, 0.11, 0.44, 0.60])
    cap = fig.text(0.05, 0.96, "", va="top", ha="left", fontsize=9.5, linespacing=1.5)

    axp.set(xlim=(-5, 5), ylim=(-5, 5), xlabel=f"$x_{d0}$", ylabel=f"$x_{d1}$", aspect="equal")
    z = landscape(task, d0, d1)
    lo_, hi_ = np.percentile(z, [1, 99.5])
    axp.imshow(z, origin="lower", extent=(-5, 5, -5, 5), cmap=SCALE, vmin=lo_, vmax=hi_,
               interpolation="nearest", zorder=0)
    axp.plot(*opt[[d0, d1]], marker="*", ms=13, mfc="white", mec="black", mew=0.6, zorder=6, ls="")
    for s_ in axp.spines.values():
        s_.set_visible(True)
        s_.set_linewidth(0.4)
    axp.set_title("Where it evaluates (2 of 10 variables)", fontsize=8.5, loc="left")
    axr.set(xlim=(0, n), ylim=(np.log10(FLOOR) - 0.5, 2.9), xlabel="Evaluations")
    axr.set_ylabel(r"Regret $\downarrow$")
    axr.set_yticks(range(-8, 3, 2))
    axr.set_yticklabels([f"$10^{{{v}}}$" for v in range(-8, 3, 2)])
    axr.set_title("Regret of the best point so far", fontsize=8.5, loc="left")
    axr.grid(axis="y", lw=0.35, color="#EBEBEB")
    for ax in (axp, axr):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def phase(i):  # which colour an evaluation belongs to
        return sum(i >= c for c in cuts)
    cols = ["#7A7A7A", *COLOURS]
    sc = axp.scatter([], [], s=16, zorder=2, linewidths=0.5)
    explore = [l == "explore_maximin" for l in labels]  # the closing check for a missed basin
    first_explore = explore.index(True) + 1 if any(explore) else None
    head, = axp.plot([], [], "o", ms=8, mfc="none", mec="white", mew=1.4, zorder=5)
    line, = axr.plot([], [], color=AGENT, lw=1.8, zorder=5)
    bl = [axr.plot([], [], color=d["colour"], lw=1.1, ls=(0, (3.2, 1.8)), zorder=3)[0]
          for d in decisions]
    bd = [axr.plot([], [], "o", ms=12, mfc="white", mec=d["colour"], mew=1.4, zorder=6)[0]
          for d in decisions]
    bt = [axr.text(0, 0, str(i + 1), ha="center", va="center", fontsize=7, fontweight="bold",
                   zorder=7, visible=False) for i in range(len(decisions))]
    lab = [axr.text(n - 4, 0, d["instead"], ha="right", va="bottom", fontsize=7, visible=False)
           for d in decisions]

    def draw(k):
        e, di, br = frames[k]
        sc.set_offsets(X[:e][:, [d0, d1]])
        fc = [cols[phase(i)] for i in range(e)]
        sc.set_facecolor(["none" if explore[i] else fc[i] for i in range(e)])
        sc.set_edgecolor([fc[i] if explore[i] else "white" for i in range(e)])
        head.set_data([X[e - 1, d0]], [X[e - 1, d1]])
        line.set_data(np.arange(1, e + 1), agent[:e])
        for j, d in enumerate(decisions):
            shown = e >= d["after"]
            bd[j].set_data([d["after"]] if shown else [], [agent[d["after"] - 1]] if shown else [])
            bt[j].set_position((d["after"], agent[d["after"] - 1]))
            bt[j].set_visible(shown)
            m = (br if di == j else (n - d["after"] if shown and (di is None or di > j) else 0)) or 0
            c = d["curve"]
            line_x = np.arange(d["after"], d["after"] + m + 1)
            bl[j].set_data(line_x, np.concatenate([[agent[d["after"] - 1]], c[d["after"]:d["after"] + m]]))
            lab[j].set_visible(m == n - d["after"] and m > 0)
            lab[j].set_position((n - 4, c[-1] + 0.2))
        if di is not None:
            d = decisions[di]
            cap.set_text(f"Decision {di + 1}, at evaluation {d['after']}\n"
                         f"Observes:  {d['observe']}\nInfers:  {d['reason']}\n"
                         f"Designs:  {d['design']}")
        elif e < cuts[0]:
            cap.set_text(f"{CAMPAIGNS[a.campaign]['task']}\nThe agent's first evaluations are a "
                         f"spread of points; it is looking for structure.")
        elif first_explore and e >= first_explore:
            cap.set_text("Converged to the optimum. The last 42 points are spread on purpose:\n"
                         "the score is the best single measurement, so a check for a missed basin "
                         "costs nothing.\n(hollow points)")
        else:
            cap.set_text(f"After decision {phase(e - 1)}: {decisions[phase(e - 1) - 1]['design']}")
        return ()

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=12), dpi=150,
              progress_callback=lambda i, n_: print(i, n_, flush=True) if i % 20 == 0 else None)
    print(a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
