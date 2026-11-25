"""Figure 2 of the paper: tasks built on common optimizers' failure modes, and how much
each method loses on them.

  python -m aobench.fig2 --runs benchmark/runs/main --agent benchmark/runs/opus_high \
      --seeds 5 --out benchmark/figures/fig2

Loss: on each instance (task, seed), how many orders of magnitude a method's final
regret is behind the best method's on that instance (log10 regret minus the lowest
log10 regret of all methods, the agent included).

(a) Each task's landscape: log10(f - f*) on a 2-D slice through the optimum of the
    seed-0 instance (coordinates 1 and 2 vary over the box, the others sit at x*),
    as observed, noise included. The 1% noise is a relative error, so it does not
    show on a log scale; the insets on the noisy tasks zoom into a step of 1e-7
    along one line, where it swamps the change a finite difference would measure.
(b) Each method's loss on each task, mean over seeds, coloured by loss (lower is
    better); a "best" stub marks a task the method was best on, on every seed, and the number is its
    worst task.
(c) Each method's loss averaged over tasks, mean +/- one standard error over seeds.
(d) Each method's loss against the fraction of the budget spent: at each fraction, its
    log10 regret minus the lowest of all methods' at that fraction, averaged over tasks
    and seeds (the agent with a band of one standard error over seeds).

SMAC and Shiwa are left out: CMA-ES beats SMAC on every task. Every method uses seeds
0..N-1 (--seeds). Scores are floored at 1e-8, as in the report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from .figures import METHODS, STYLE, WIDTH, load  # noqa: E402
from .tasks import SPECS, make_task  # noqa: E402

TASKS = ["ill_conditioned", "ill_conditioned_noisy", "noise_dominated", "plateaus", "few_evals_10d",
         "unstructured_2d"]
SHORT = {
    "ill_conditioned": "Ill-conditioned",
    "ill_conditioned_noisy": "Ill-cond., noisy",
    "noise_dominated": "Noise-dominated",
    "plateaus": "Plateaus",
    "few_evals_10d": "Low budget, 10-D",
    "unstructured_2d": "Unstructured, low budget",
}
LANDSCAPE = {  # function, and its condition number or noise; kappa = condition number
    "ill_conditioned": r"Ellipsoid, $\kappa = 10^6$",
    "ill_conditioned_noisy": r"Ellipsoid, $\kappa = 10^6$, 1% noise",
    "noise_dominated": r"Sphere, 100% noise",
    "plateaus": r"Step ellipsoid, $\kappa = 10^2$",
    "few_evals_10d": r"Sphere, 1% noise",
    "unstructured_2d": r"Gallagher, 101 peaks",
}
OPTIMIZERS = ["lbfgsb", "cobyla", "cma_es", "gpbo", "bandit"]  # panel b, before the agent
METHODS_C = OPTIMIZERS + ["sobol", "agent"]
AGENT = METHODS["agent"][1]
AGENT_TINT = "#F4EDF7"
BAR = "#9AA3AD"
BAR_WORST = "#4A525A"
# One scale for panels a and b: both show orders of magnitude above the best value, so a colour
# means the same number of decades in either panel. It runs light and warm (at the best) to dark
# (far behind), truncated so neither end is paper-white or ink-black; perceptually uniform, and
# it survives greyscale printing.
SCALE = LinearSegmentedColormap.from_list("scale", plt.get_cmap("magma_r")(np.linspace(0.10, 0.90, 256)))
LOSS_CMAP = SCALE
LOSS_NORM = Normalize(0, 10)
# Panel b: this method was the best on this task. The bottom of the scale, so a win is drawn
# in the colour panel a gives the optimum.
BEST = SCALE(0.0)
YMAX = 12.0


def landscape(task_key: str, n: int = 161):
    task = make_task(task_key, seed=0)
    xo = np.asarray(task._problem.optimum.x, dtype=float)
    g = np.linspace(-5, 5, n)
    z = np.empty((n, n))
    for i, b in enumerate(g):
        for j, a in enumerate(g):
            x = xo.copy()
            x[0], x[1] = a, b
            z[i, j] = task.evaluate(x).y - task.f_opt
    return np.log10(np.maximum(z, 1e-8)), xo, task


def noise_inset(ax, task, box=(0.58, 0.15, 0.38, 0.26)):
    """Observed vs true value along a 1e-7 step from a point near the optimum."""
    xo = np.asarray(task._problem.optimum.x, dtype=float)
    x0 = xo + 0.5
    u = np.ones_like(xo) / np.sqrt(len(xo))
    t = np.linspace(0, 1e-7, 40)
    true = np.array([task._problem(x0 + s * u) for s in t]) - task.f_opt
    seen = np.array([task.evaluate(x0 + s * u).y for s in t]) - task.f_opt
    ins = ax.inset_axes(list(box))
    ins.set_facecolor("white")
    ins.plot(t, seen / true[0] - 1, ".", ms=1.6, color="0.45")
    ins.plot(t, true / true[0] - 1, "-", lw=0.9, color=AGENT)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
    # Under the inset and right-aligned on its edge: clear of the optimum marker at the centre
    # of every panel, and inside the panel whatever the label's width. White, because down there
    # it sits over the dark end of the scale.
    ax.text(box[0] + box[2], box[1] - 0.025, r"zoom: step $10^{-7}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=5, color="white")


def panel_a(fig, gs):
    axes = []
    for k, t in enumerate(TASKS):
        ax = fig.add_subplot(gs[k])
        z, xo, task = landscape(t)
        lo, hi = np.percentile(z, [1, 99.5])
        ax.imshow(z, origin="lower", extent=(-5, 5, -5, 5), cmap=SCALE, vmin=lo, vmax=hi,
                  interpolation="nearest", rasterized=True)
        ax.plot(xo[0], xo[1], marker="*", ms=7, mfc="white", mec="black", mew=0.5)
        if t in ("ill_conditioned_noisy", "few_evals_10d"):
            noise_inset(ax, task)
        spec = SPECS[t]
        name = SHORT[t].replace(", low budget", ",\nlow budget") if len(SHORT[t]) > 18 else SHORT[t]
        ax.set_title(f"$\\mathbf{{{k + 1}}}$  {name}", pad=9, fontsize=7.5, linespacing=1.0)
        ax.text(0.5, 1.015, LANDSCAPE[t], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6, color="0.3")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.4)
        ax.set_xlabel(f"$d={spec.dim}$, {spec.budget} Evals", fontsize=6.5, labelpad=2)
        axes.append(ax)
    return axes


def losses(data):
    """method -> (task, seed) array of orders of magnitude behind the best method on that instance."""
    ms = METHODS_C
    L = np.array([[data[t][m][:, -1] for m in ms] for t in TASKS])  # task, method, seed
    loss = L - L.min(1, keepdims=True)
    return {m: loss[:, i, :] for i, m in enumerate(ms)}


def panel_b(fig, gs, loss):
    axes = []
    cells = OPTIMIZERS + ["agent"]
    x = np.arange(1, len(TASKS) + 1)
    for k, m in enumerate(cells):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        v = loss[m].mean(1)
        worst = int(np.argmax(v))
        agent = m == "agent"
        dark = v >= v[worst] - 0.05  # ties for the worst task are all labelled
        colours = LOSS_CMAP(LOSS_NORM(v))
        if agent:
            ax.set_facecolor(AGENT_TINT)
        ax.bar(x, v, width=0.72, color=colours, zorder=2)
        best = v < 0.15  # best, or within a rounding of it: a win, drawn as a stub so it is visible
        ax.bar(x[best], np.full(best.sum(), 0.45), width=0.72, color=BEST, zorder=3)
        for i in np.flatnonzero(best):  # say what the stub means, so the figure needs no key
            ax.text(x[i], 0.8, "best", ha="center", va="bottom", fontsize=6, color="black")
        for i in np.flatnonzero(dark):
            ax.text(x[i], v[i] + 0.35, f"{v[i]:.1f}", ha="center", va="bottom", fontsize=6.5,
                    color=AGENT if agent else "0.25", fontweight="bold" if agent else "normal")
        ax.set_ylim(0, YMAX)
        ax.set_xlim(0.4, len(TASKS) + 0.6)
        ax.set_yticks([0, 5, 10])
        ax.set_xticks(x)
        ax.tick_params(length=0, pad=1.5, labelsize=6)
        if k % 3:
            ax.set_yticklabels([])
        ax.grid(axis="y", lw=0.3, color="0.85", zorder=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.set_title(METHODS[m][0], fontsize=7.5, pad=5, color=AGENT if agent else "black",
                     fontweight="bold" if agent else "normal")
        axes.append(ax)
    return axes


def panel_c(ax, loss):
    n = loss["agent"].shape[1]
    per_seed = {m: loss[m].mean(0) for m in METHODS_C}  # average over tasks, per seed
    mean = {m: v.mean() for m, v in per_seed.items()}
    se = {m: v.std(ddof=1) / np.sqrt(n) for m, v in per_seed.items()}
    order = sorted(METHODS_C, key=mean.get)
    for y, m in enumerate(order):
        agent = m == "agent"
        colour = AGENT if agent else ("#D5D9DD" if m == "sobol" else BAR)
        ax.barh(y, mean[m], height=0.66, color=colour, zorder=2)
        ax.errorbar(mean[m], y, xerr=se[m], fmt="none", ecolor="0.25", elinewidth=0.7, capsize=1.6,
                    capthick=0.6, zorder=3)
        ax.text(mean[m] + se[m] + 0.2, y, f"{mean[m]:.1f}", va="center", fontsize=6.5,
                color=AGENT if agent else "0.25", fontweight="bold" if agent else "normal")
    labels = [METHODS[m][0] for m in order]
    ax.set_yticks(range(len(order)), labels)
    for lab, m in zip(ax.get_yticklabels(), order):
        if m == "agent":
            lab.set_color(AGENT)
            lab.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 9.5)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.3, color="0.85", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Final Log$_{10}$ Regret Ratio")
    return mean, se


FRACS = np.linspace(0.01, 1, 100)  # panel d: fractions of each task's budget


def anytime_losses(data):
    """method -> (fraction, seed) log10 regret ratio after that fraction of the budget, mean over tasks.

    At each fraction, each method's log10 regret on an instance minus the lowest of all methods' at
    the same fraction: the ratio of regrets at equal spend, as panel c's is at the end."""
    out = {m: [] for m in METHODS_C}
    for t in TASKS:
        budget = SPECS[t].budget
        idx = np.ceil(FRACS * budget).astype(int) - 1
        L = np.array([data[t][m][:, idx] for m in METHODS_C])  # method, seed, fraction
        loss = L - L.min(0, keepdims=True)
        for i, m in enumerate(METHODS_C):
            out[m].append(loss[i].T)  # fraction, seed
    return {m: np.mean(v, axis=0) for m, v in out.items()}


def panel_d(ax, anytime):
    """Each method's log10 regret ratio against the fraction of the budget spent."""
    order = sorted(METHODS_C, key=lambda m: anytime[m][-1].mean())
    n = anytime["agent"].shape[1]
    ends = []
    for m in reversed(order):  # agent drawn last, on top
        name, colour, _ = METHODS[m]
        agent = m == "agent"
        mean, se = anytime[m].mean(1), anytime[m].std(1, ddof=1) / np.sqrt(n)
        if agent:
            ax.fill_between(FRACS, mean - se, mean + se, color=colour, alpha=0.15, lw=0, zorder=3)
        ax.plot(FRACS, mean, color=colour, lw=2.0 if agent else 1.0, ls=":" if m == "sobol" else "-",
                zorder=4 if agent else 2)
        ends.append([mean[-1], name, colour, agent])
    # Labels at the right end, nudged apart so they do not overlap.
    ends.sort(key=lambda e: e[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.62)
    for (_, name, colour, agent), y in zip(ends, ys):
        ax.text(1.02, y, name, color=colour, fontsize=6.3, va="center", ha="left",
                fontweight="bold" if agent else "normal", clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 9.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.grid(lw=0.3, color="0.88", zorder=0)
    ax.set_xlabel("Fraction of Budget")
    ax.set_ylabel(r"Log$_{10}$ Regret Ratio $\downarrow$")
    return {m: anytime[m].mean(1) for m in METHODS_C}


def fit_row(axes, x0, x1, box):
    """Squeeze a row of axes into [x0, x1], keeping their relative widths and gaps."""
    pos = [ax.get_position() for ax in axes]
    a, b = pos[0].x0, pos[-1].x1
    s = (x1 - x0) / (b - a)
    for ax, p in zip(axes, pos):
        ax.set_position([x0 + (p.x0 - a) * s, p.y0, p.width * s, p.height])


def draw(loss, anytime, path: Path, gap: float = 0.055):
    fig = plt.figure(figsize=(WIDTH, 7.0))
    # Row a's height is set so the square landscapes fill it; b and c/d get the rest.
    outer = GridSpec(3, 1, figure=fig, height_ratios=[0.90, 2.10, 1.75], hspace=0.40, left=0.02,
                     right=0.98, top=0.96, bottom=0.06)
    top = outer[0].subgridspec(1, len(TASKS), wspace=0.22)
    grid_b = outer[1].subgridspec(2, 3, wspace=0.11, hspace=0.45)
    bottom = outer[2].subgridspec(1, 2, wspace=0.3)
    axa = panel_a(fig, top)
    axb = panel_b(fig, grid_b, loss)
    axc = fig.add_subplot(bottom[0])
    axd = fig.add_subplot(bottom[1])
    mean, se = panel_c(axc, loss)
    curve = panel_d(axd, anytime)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    box = lambda ax: inv.transform(ax.get_tightbbox(r).get_points())
    left, right = box(axa[0])[0, 0], box(axa[-1])[1, 0]
    # One y-axis label for panel b's six subplots, at a's left edge; the row makes room for it.
    b0, b3 = axb[0].get_position(), axb[3].get_position()
    ylab = fig.text(left, (b0.y1 + b3.y0) / 2, r"Log$_{10}$ Regret Ratio $\downarrow$", rotation=90,
                    va="center", ha="left", fontsize=7)
    fig.canvas.draw()
    b_left = inv.transform(ylab.get_window_extent(r).get_points())[1, 0] + 0.012
    b_left += axb[0].get_position().x0 - box(axb[0])[0, 0]  # room for the y tick labels too
    for row in (axb[:3], axb[3:]):
        fit_row(row, b_left, right, box)
    fig.canvas.draw()
    # c and d share the bottom row: tight boxes from a's left edge to its right, one gap between.
    (cl, _), (cr, _) = box(axc)
    (dl, _), (dr, _) = box(axd)
    pc, pd = axc.get_position(), axd.get_position()
    pad_c, pad_d = (pc.x0 - cl, cr - pc.x1), (pd.x0 - dl, dr - pd.x1)
    room = right - left - gap - sum(pad_c) - sum(pad_d)
    wc, wd = 0.45 * room, 0.55 * room
    axc.set_position([left + pad_c[0], pc.y0, wc, pc.height])
    axd.set_position([left + pad_c[0] + wc + pad_c[1] + gap + pad_d[0], pd.y0, wd, pd.height])

    fig.canvas.draw()
    # One "Task" label for panel b's six subplots, centred under its bottom row.
    bx0, bx1 = axb[3].get_position().x0, axb[5].get_position().x1
    fig.text((bx0 + bx1) / 2, min(box(ax)[0, 1] for ax in axb[3:]) - 0.010, "Task",
             ha="center", va="top", fontsize=7)
    fig.canvas.draw()
    at = box(axa[0])[1, 1]
    bt = max(box(ax)[1, 1] for ax in axb[:3])
    ct = max(box(axc)[1, 1], box(axd)[1, 1])
    for x, y, letter in ((left, at, "a"), (left, bt, "b"), (box(axc)[0, 0], ct, "c"),
                         (box(axd)[0, 0], ct, "d")):
        fig.text(x - 0.012, y + 0.006, letter, fontsize=11, fontweight="bold", va="bottom", ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    return mean, se, curve


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m aobench.fig2", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("benchmark/runs/main"), help="baseline run folder")
    p.add_argument("--agent", type=Path, default=Path("benchmark/runs/opus_high"), help="agent run folder")
    p.add_argument("--seeds", type=int, default=5, help="use seeds 0..N-1 for every method")
    p.add_argument("--out", type=Path, default=Path("benchmark/figures/fig2"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    data = load([a.runs, a.agent], a.seeds)
    for t in TASKS:
        n = {m: len(data[t][m]) for m in METHODS_C}
        assert len(set(n.values())) == 1, f"{t}: uneven seeds {n}"
    loss = losses(data)
    mean, se, curve = draw(loss, anytime_losses(data), a.out / "fig2")
    print("log10 regret ratio per task (mean over seeds) | average +/- s.e. | at 10%, 25%, 50% of budget:")
    for m in METHODS_C:
        print(f"  {METHODS[m][0]:9s}" + " ".join(f"{v:5.1f}" for v in loss[m].mean(1))
              + f" | {mean[m]:.2f} +/- {se[m]:.2f} | " + " ".join(f"{curve[m][i]:.2f}" for i in (9, 24, 49)))
    print(f"wrote {a.out}/fig2 (.pdf and .png)")


if __name__ == "__main__":
    main()
