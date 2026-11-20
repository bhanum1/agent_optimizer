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
    show on a log scale; the inset on the noisy ellipsoid zooms into a step of 1e-7
    along one line, where it swamps the change a finite difference would measure.
(b) Each method's loss on each task, mean over seeds, coloured by loss (lower is
    better); the number is its worst task.
(c) fig2_bars: each method's loss averaged over tasks, mean +/- one standard error
    over seeds. fig2_profile: a performance profile (Dolan & More 2002), the
    fraction of runs whose loss is at most tau.

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
# Panel b: bar colour follows loss, light (near the best) to deep red (far behind).
LOSS_CMAP = LinearSegmentedColormap.from_list("loss", ["#D9DEE3", "#F6C3A5", "#E0694A", "#A3201C", "#5C0A0E"])
LOSS_NORM = Normalize(0, 10)
YMAX = 12.5


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


def noise_inset(ax, task):
    """Observed vs true value along a 1e-7 step from a point near the optimum."""
    xo = np.asarray(task._problem.optimum.x, dtype=float)
    x0 = xo + 0.5
    u = np.ones_like(xo) / np.sqrt(len(xo))
    t = np.linspace(0, 1e-7, 40)
    true = np.array([task._problem(x0 + s * u) for s in t]) - task.f_opt
    seen = np.array([task.evaluate(x0 + s * u).y for s in t]) - task.f_opt
    ins = ax.inset_axes([0.5, 0.04, 0.46, 0.4])
    ins.set_facecolor("white")
    ins.plot(t, seen / true[0] - 1, ".", ms=1.6, color="0.45")
    ins.plot(t, true / true[0] - 1, "-", lw=0.9, color=AGENT)
    ins.set_xticks([])
    ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
    ins.set_title(r"zoom: step $10^{-7}$", fontsize=5, pad=1.2)


def panel_a(fig, gs):
    axes = []
    for k, t in enumerate(TASKS):
        ax = fig.add_subplot(gs[k])
        z, xo, task = landscape(t)
        lo, hi = np.percentile(z, [1, 99.5])
        ax.imshow(z, origin="lower", extent=(-5, 5, -5, 5), cmap="viridis", vmin=lo, vmax=hi,
                  interpolation="nearest", rasterized=True)
        ax.plot(xo[0], xo[1], marker="*", ms=7, mfc="white", mec="black", mew=0.5)
        if t == "ill_conditioned_noisy":
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
        ax.set_xlabel(f"$d={spec.dim}$, {spec.budget} evals", fontsize=6.5, labelpad=2)
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
        ax.plot(x[v < 0.15], np.full((v < 0.15).sum(), 0.25), "_", color="0.55", ms=4, mew=1, zorder=3)  # zero-height bars
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
    ax.set_xlabel(r"$\log_{10}$ regret ratio (mean over tasks)")
    return mean, se


def panel_c_profile(ax, loss):
    """Performance profile (Dolan & More 2002): fraction of instances within tau of the best."""
    tau = np.linspace(0, 11, 1101)
    order = sorted(METHODS_C, key=lambda m: loss[m].mean())
    out = {}
    for m in reversed(order):  # agent drawn last, on top
        v = loss[m].ravel()
        frac = (v[None, :] <= tau[:, None] + 1e-9).mean(1)
        name, colour, _ = METHODS[m]
        agent = m == "agent"
        ax.step(tau, frac, where="post", color=colour, lw=2.0 if agent else 1.0,
                ls=":" if m == "sobol" else "-", zorder=4 if agent else 2, label=name)
        out[m] = (frac[0], frac[100])  # at tau = 0 and tau = 1
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
    ax.grid(lw=0.3, color="0.88", zorder=0)
    ax.set_xlabel(r"$\log_{10}$ regret ratio, $\tau$")
    ax.set_ylabel(r"Fraction of runs within $\tau$")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], loc="lower right", frameon=True, fontsize=6.5, handlelength=1.6,
                    labelspacing=0.3, borderpad=0.4, framealpha=0.95)
    leg.get_frame().set(facecolor="white", edgecolor="none")
    leg.set_zorder(5)
    for lab in ax.get_legend().get_texts():
        if lab.get_text() == METHODS["agent"][0]:
            lab.set_color(AGENT)
            lab.set_fontweight("bold")
    return out


def colour_bar(fig, axb, box):
    """Horizontal key for panel b's bar colours, centred under its bottom row."""
    x0, x1 = axb[3].get_position().x0, axb[5].get_position().x1
    w = 0.45 * (x1 - x0)
    y = box(axb[3])[0, 1] - 0.035
    cb = fig.add_axes([(x0 + x1 - w) / 2, y, w, 0.014])
    cb.imshow(np.linspace(0, 10, 256)[None, :], aspect="auto", cmap=LOSS_CMAP, norm=LOSS_NORM,
              extent=(0, 10, 0, 1))
    cb.set_yticks([])
    cb.set_xticks([0, 5, 10])
    cb.tick_params(length=0, labelsize=6, pad=1.5)
    for sp in cb.spines.values():
        sp.set_visible(False)
    cb.text(-0.4, 0.5, "best", ha="right", va="center", fontsize=6, color="0.3")
    cb.text(10.4, 0.5, "worse", ha="left", va="center", fontsize=6, color="0.3")


def draw(loss, variant: str, path: Path, gap: float = 0.075):
    fig = plt.figure(figsize=(WIDTH, 4.9))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1.55], hspace=0.3, left=0.02, right=0.98,
                     top=0.93, bottom=0.14)
    top = outer[0].subgridspec(1, len(TASKS), wspace=0.22)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1], wspace=0.3)
    grid_b = bottom[0].subgridspec(2, 3, wspace=0.14, hspace=0.55)
    axa = panel_a(fig, top)
    axb = panel_b(fig, grid_b, loss)
    axc = fig.add_subplot(bottom[1])
    summary = panel_c_profile(axc, loss) if variant == "profile" else panel_c(axc, loss)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    box = lambda ax: inv.transform(ax.get_tightbbox(r).get_points())
    left, right = box(axa[0])[0, 0], box(axa[-1])[1, 0]
    b0, b3 = axb[0].get_position(), axb[3].get_position()
    ylab = fig.text(left, (b0.y1 + b3.y0) / 2, r"$\log_{10}$ regret ratio", rotation=90, va="center",
                    ha="left", fontsize=7)
    fig.canvas.draw()
    shift = inv.transform(ylab.get_window_extent(r).get_points())[1, 0] + 0.012 - box(axb[0])[0, 0]
    for ax in axb:
        ax.set_position(ax.get_position().translated(shift, 0))
    fig.canvas.draw()
    b_right = max(box(ax)[1, 0] for ax in axb)
    pc = axc.get_position()
    (cl, _), (cr, _) = box(axc)
    x0, x1 = pc.x0 + (b_right + gap) - cl, pc.x1 + right - cr  # c fills from one gap after b to a's edge
    axc.set_position([x0, b3.y0, x1 - x0, b0.y1 - b3.y0])
    fig.canvas.draw()
    colour_bar(fig, axb, box)
    at = box(axa[0])[1, 1]
    bt = max(box(axb[0])[1, 1], box(axc)[1, 1])
    for x, y, letter in ((left, at, "a"), (left, bt, "b"), (box(axc)[0, 0], bt, "c")):
        fig.text(x - 0.012, y + 0.008, letter, fontsize=11, fontweight="bold", va="bottom", ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    return summary


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
    mean, se = draw(loss, "bars", a.out / "fig2_bars")
    prof = draw(loss, "profile", a.out / "fig2_profile")
    print("log10 regret ratio per task (mean over seeds) | average +/- s.e. | best on, within 1 on:")
    for m in METHODS_C:
        print(f"  {METHODS[m][0]:9s}" + " ".join(f"{v:5.1f}" for v in loss[m].mean(1))
              + f" | {mean[m]:.2f} +/- {se[m]:.2f} | {prof[m][0]:.0%}, {prof[m][1]:.0%}")
    print(f"wrote {a.out}/fig2_bars and fig2_profile (.pdf and .png)")


if __name__ == "__main__":
    main()
