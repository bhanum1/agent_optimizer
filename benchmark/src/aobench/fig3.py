"""Figure 3 of the paper: one campaign's key decisions, and what each one bought.

  python -m aobench.fig3 --campaign few_evals_10d/1 --runs benchmark/runs/opus_high \
      --baselines benchmark/runs/main --branches benchmark/runs/counterfactual --out benchmark/figures/fig3

(a) The task as the agent saw it, and its key decisions: the evidence, its reasoning and
    the decision, from its rationales (decisions.md), with what each bought: the agent's
    final regret against its counterfactual's.
(b) log10 regret against evaluations. Solid: the agent's campaign. Dashed: each
    counterfactual, what would have run had the agent not made that decision: the campaign
    rewound to it and the block before it carried on to the end of the budget (the
    agent's saved optimizer, or its script restarted on the same data;
    aobench.counterfactual). Grey: the best baseline's whole run on the same instance
    (best by final regret).

Branches each campaign needs are listed in CAMPAIGNS (folder names under
--branches/<task>/seed_<seed>/). For few_evals_10d/1:

  python -m aobench.counterfactual --runs benchmark/runs/opus_high --task few_evals_10d --seed 1 \
      --after 47 --proposal '{"script": "work/bo.py", "label": "gpbo", "args": {"seed": 1, "n_init": 0, "local_frac": 0.5}, "state": "restart"}'
  python -m aobench.counterfactual --runs benchmark/runs/opus_high --task few_evals_10d --seed 1 \
      --after 126 --proposal '{"script": "work/lq.py", "label": "lq", "args": {}, "state": "continue"}'
"""
from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from matplotlib.patches import FancyBboxPatch  # noqa: E402

from .counterfactual import workspace  # noqa: E402
from .figures import METHODS, STYLE, WIDTH  # noqa: E402
from .tasks import make_task  # noqa: E402

FLOOR = 1e-8
AGENT = METHODS["agent"][1]
BASELINES = ["lbfgsb", "cobyla", "cma_es", "gpbo", "bandit", "sobol"]
BOLD = dict(fontfamily="STIXGeneral", fontweight="bold")  # STIX Two Text has no bold weight here
# All text is black: colour is left to the curves, the badge rings and the stems, and every
# label sits against the thing it names, so no label needs its colour to be found.
INK = MUTED = FAINT = "black"
RULE = "#D9D9D9"
GREY = "#9A9A9A"  # the baseline
COLOURS = ["#D55E00", "#0072B2", "#009E73"]  # Okabe-Ito vermillion, blue, green: one per decision

# The task as shape, function, budget and noise, and each key decision: when, the decision and why
# (from its rationale in decisions.md), its counterfactual branch, and that branch's label in
# panel b (what runs instead), as the three steps of the agent's loop. `label_at` places a
# label by hand.
CAMPAIGNS = {
    "few_evals_10d/1": dict(
        task="10-dimensional black box with 1% noise. 300 evaluation budget.",
        decisions=[
            dict(after=47, branch="gpbo_47", instead="GP-BO continued",
                 observe="A separable quadratic fits all 47 points.",
                 reason="Its minimum can be computed, not searched.",
                 design="Jump to the fitted minimum, then refine locally."),
            dict(after=126, branch="lq_126", instead="Trust region continued", label_at=(256, -5.72),
                 observe=r"The trust region stalled; repeats agree to $5{\times}10^{-7}$.",
                 reason="Noise is negligible, so slopes can be measured.",
                 design="Write its own finite-difference Newton method."),
        ],
        agent_label=(104, -5.1), base_label=(160, 0.6)),
}


def curve(path: Path, task) -> np.ndarray:
    """log10 regret of the best measured point after each evaluation, from a data.csv."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "measured" and r["y"] != ""]
    best, regret, out = np.inf, np.inf, []
    for r in rows[:task.spec.budget]:
        y = float(r["y"])
        if y < best:
            x = np.array([float(r[f"x{i}"]) for i in range(task.dim)])
            best, regret = y, float(task._problem(x)) - task.f_opt
        out.append(np.log10(max(regret, FLOOR)))
    return np.array(out)


def trace(path: Path) -> np.ndarray:
    """log10 regret after each evaluation, from a run's result CSV."""
    with open(path, newline="") as f:
        return np.log10(np.maximum([float(x["regret"]) for x in csv.DictReader(f)], FLOOR))


def verdict(gain: float) -> str:
    """What a decision bought, from the gap in final log10 regret to its counterfactual."""
    if abs(gain) < 0.3:
        return "no change"
    k = round(abs(gain))
    return rf"$10^{{{k}}}\times$ {'less' if gain > 0 else 'more'} regret"


def load(name: str, runs: Path, baselines: Path, branches: Path):
    task_key, seed = name.split("/")
    seed = int(seed)
    task = make_task(task_key, seed)
    budget = task.spec.budget
    agent = curve(workspace(runs, task_key, seed) / ".agentoptimizer" / "data.csv", task)
    runs_b = {m: trace(baselines / task_key / m / f"seed_{seed}.csv") for m in BASELINES}
    best = min(runs_b, key=lambda m: runs_b[m][-1])
    decisions = []
    for i, d in enumerate(CAMPAIGNS[name]["decisions"]):
        path = branches / task_key / f"seed_{seed}" / d["branch"] / "r0" / "workspace" / ".agentoptimizer" / "data.csv"
        c = curve(path, task)
        c = np.concatenate([c, np.full(budget - len(c), c[-1])]) if len(c) < budget else c  # stopped early
        decisions.append({**d, "colour": COLOURS[i], "curve": c, "gain": c[-1] - agent[-1]})
    return task, agent, (best, runs_b[best]), decisions


def badge(ax, x, y, number, colour, size=11.5, transform=None):
    """A numbered ring: the same mark in both panels, so a decision reads across them."""
    kw = {"transform": transform} if transform is not None else {}
    ax.plot([x], [y], "o", ms=size, mfc="white", mec=colour, mew=1.3, zorder=6, clip_on=False, **kw)
    ax.text(x, y, str(number), ha="center", va="center", fontsize=size * 0.6, color="black", zorder=7,
            **kw, **BOLD)


def rendered_len(text: str) -> int:
    """Length of a string as it is drawn: a $...$ run sets about a third of its source."""
    out, math = 0, False
    for part in text.split("$"):
        out += round(len(part) / 3) if math else len(part)
        math = not math
    return out


def run(fig, ax, x, y, parts, size, gap=3.0):
    """A left-to-right run of differently styled pieces, laid out as one right-aligned block."""
    tr = ax.transAxes
    r = fig.canvas.get_renderer()
    inv = tr.inverted()
    g = gap / (ax.get_position().width * fig.get_figwidth() * 72)
    objs, cx = [], 0.0
    for text, font in parts:
        t = ax.text(cx, y, text, ha="left", va="top", fontsize=size, color=INK, transform=tr, **font)
        box = inv.transform(t.get_window_extent(r).get_points())
        objs.append(t)
        cx += box[1, 0] - box[0, 0] + g
    for t in objs:
        t.set_x(t.get_position()[0] + x - (cx - g))


HIGHLIGHT = "#FCF4D4"  # the shading behind a step Figure 4 wants read as the point of the card
HIGHLIGHT_INK = "#9A7B1A"


def cards(fig, ax, task_text, decisions, wrap=58, steps=("Observe", "Reason", "Design"),
          highlight=None, highlight_label=None):
    """One card per decision. Figure 4 draws its panels with this too, so they match by sharing it.

    Typeset in points from the top of the panel, so spacing stays even at any figure size.
    Each decision is one turn of the agent's loop, read down a stem with a stop at each step:
    what it observed (roman), what it inferred (italic), the optimizer it designed (bold). The three
    steps are named once, above the first card, instead of labelling all six lines; `steps`
    renames them, which is how Figure 4 says what its middle step reasons from. `highlight`
    names a step to shade in every card, and `highlight_label` names the shading once, over
    the first band: Figure 4 uses the two to say that the middle step is the domain knowledge
    and that the domain knowledge is the point. Every decision needs `observe`, `reason`,
    `design`, `colour`, and the two footers `foot` and `bought`.
    """
    ax.set_axis_off()
    h_pt = ax.get_position().height * fig.get_figheight() * 72
    y_of = lambda pt: 1 - pt / h_pt  # noqa: E731  distance from the top, in points, to axes units
    tr = ax.transAxes
    x_stem, x_text = 0.022, 0.085
    wrapped = lambda s, w: textwrap.wrap(s, w + len(s) - rendered_len(s))  # noqa: E731

    ax.text(0.0, y_of(1), task_text, ha="left", va="top", fontsize=7.6, color=INK, transform=tr)
    ax.plot([0, 1], [y_of(15)] * 2, color=RULE, lw=0.6, transform=tr, clip_on=False)
    arrow = (r"$\rightarrow$", {})  # STIX Two Text has no arrow glyph of its own
    run(fig, ax, 1.0, y_of(19.5), [(steps[0], {}), arrow, (steps[1], {"style": "italic"}), arrow,
                                   (steps[2], BOLD)], 6.2)

    # (key, size, line height, extra font kwargs) -- one block per step of the loop.
    STEPS = (("observe", 6.9, 9.6, {}), ("reason", 6.9, 9.6, {"style": "italic"}),
             ("design", 7.5, 10.6, BOLD))
    t, gap = 36, 18
    bands, tag = [], []
    for i, d in enumerate(decisions):
        stops = []
        for key, size, lead, font in STEPS:
            lines = wrapped(d[key], round(wrap * 6.9 / size))
            stops.append(t + 1)
            top = t - 2.2
            if key == highlight and highlight_label:
                # The tag gets a line of the band to itself: the step's own text is as long
                # as the card is wide, so there is no room for it beside the words.
                tag.append((top, t + 1.6))
                t += 10.5
            for line in lines:
                ax.text(x_text, y_of(t), line, ha="left", va="top", fontsize=size, color=INK,
                        transform=tr, **font)
                t += lead
            if key == highlight:
                bands.append((top, t - lead + size + 1.4))
            t += 2.6
        foot = t + 1.5
        ax.text(x_text, y_of(foot), d["foot"], ha="left", va="top", fontsize=6.4, color=FAINT,
                transform=tr)
        ax.text(1.0, y_of(foot - 0.6), d["bought"], ha="right", va="top", fontsize=7.5,
                color="black", transform=tr)
        badge(ax, x_stem, y_of(stops[0] + 2.6), i + 1, d["colour"], size=9.5, transform=tr)
        ax.plot([x_stem] * 2, [y_of(stops[0] + 9), y_of(foot + 6)], color=d["colour"], lw=1.1,
                alpha=0.45, transform=tr, clip_on=False, solid_capstyle="butt")
        for y in stops[1:]:  # a stop on the stem beside each later step
            ax.plot([x_stem], [y_of(y + 2.6)], "o", ms=2.6, color=d["colour"], transform=tr,
                    clip_on=False, zorder=6)
        t = foot + 6 + gap

    # The shaded step, under the text it belongs to, tagged at the right of every band.
    for top, bot in bands:
        y, h = y_of(bot), y_of(top) - y_of(bot)
        ax.add_patch(FancyBboxPatch((x_text - 0.018, y), 1.018 - x_text, h, transform=tr,
                                    boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=4,
                                    facecolor=HIGHLIGHT, edgecolor="none", zorder=0,
                                    clip_on=False))
    for _, at in tag:  # at the top right of its band, on a line of its own
        ax.text(0.985, y_of(at), highlight_label.upper(), ha="right", va="top", fontsize=5.2,
                color=HIGHLIGHT_INK, transform=tr, clip_on=False, **BOLD)


def panel_b(ax, name, task, agent, base, decisions):
    budget, lo = task.spec.budget, np.log10(FLOOR)
    cfg = CAMPAIGNS[name]
    best, b = base
    ax.grid(axis="y", lw=0.35, color="#EBEBEB", zorder=0)
    ax.plot(np.arange(1, len(b) + 1), b, color=GREY, lw=1.0, zorder=2)
    ax.text(*cfg["base_label"], f"{METHODS[best][0]}, best baseline", fontsize=6.5, color="black",
            ha="left", va="top")
    x = np.arange(1, budget + 1)
    for d in decisions:
        c, k = d["curve"], d["after"]
        ax.plot(x[k - 1:], c[k - 1:], color=d["colour"], lw=1.0, ls=(0, (3.2, 1.8)), zorder=3)
        ax.plot([budget], [c[-1]], "o", ms=3, color=d["colour"], zorder=4, clip_on=False)
        lx, ly = d.get("label_at", (budget - budget / 60, c[-1] + 0.25))
        ax.text(lx, ly, d["instead"], ha="right", va="bottom", fontsize=6.5, color="black")
    ax.plot(np.arange(1, len(agent) + 1), agent, color=AGENT, lw=1.7, zorder=5, solid_joinstyle="miter")
    ax.text(*cfg["agent_label"], "Agent", color="black", fontsize=7.2, ha="right", va="center", **BOLD)
    for i, d in enumerate(decisions):
        badge(ax, d["after"], agent[d["after"] - 1], i + 1, d["colour"], size=10)
    top = np.ceil(max(agent.max(), b.max()))
    ax.set_xlim(0, budget)
    ax.set_ylim(lo - 0.5, top + 0.6)
    ax.set_yticks(np.arange(-8, top + 1, 2))
    ax.set_yticklabels([f"$10^{{{int(v)}}}$" for v in np.arange(-8, top + 1, 2)])
    ax.set_xlabel("Evaluations")
    ax.set_ylabel(r"Regret $\downarrow$")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--campaign", default="few_evals_10d/1", choices=sorted(CAMPAIGNS))
    p.add_argument("--runs", type=Path, default=Path("benchmark/runs/opus_high"))
    p.add_argument("--baselines", type=Path, default=Path("benchmark/runs/main"))
    p.add_argument("--branches", type=Path, default=Path("benchmark/runs/counterfactual"))
    p.add_argument("--out", type=Path, default=Path("benchmark/figures/fig3"))
    a = p.parse_args(argv)
    task, agent, base, decisions = load(a.campaign, a.runs, a.baselines, a.branches)
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(WIDTH, 2.72))
    ax_a = fig.add_axes([0.035, 0.05, 0.44, 0.83])
    ax_b = fig.add_axes([0.585, 0.17, 0.405, 0.68])
    cards(fig, ax_a, CAMPAIGNS[a.campaign]["task"],
          [{**d, "foot": f"eval {d['after']}", "bought": verdict(d["gain"])} for d in decisions])
    panel_b(ax_b, a.campaign, task, agent, base, decisions)
    for ax, letter, title in ((ax_a, "a", "Agent's Key Decisions and Reasoning"),
                              (ax_b, "b", "Benefits of Each Key Decision")):
        x0 = 0.0 if letter == "a" else ax.get_position().x0 - 0.065
        fig.text(x0, 0.975, letter, fontsize=11, va="top", ha="left", **BOLD)
        fig.text(x0 + 0.025, 0.968, title, fontsize=8, va="top", ha="left", color=INK)
    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"fig3.{ext}")
    print(a.out / "fig3.png")
    print(f"agent final {agent[-1]:.2f}; best baseline {METHODS[base[0]][0]} {base[1][-1]:.2f}")
    for d in decisions:
        print(f"  eval {d['after']}: {d['instead']} ends at {d['curve'][-1]:.2f} ({d['gain']:+.2f} decades)")


if __name__ == "__main__":
    main()
