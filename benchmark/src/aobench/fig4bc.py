"""Figure 4, panel b: one real campaign, in the form of Figure 3.

    .venv/bin/python -m aobench.fig4bc --out benchmark/figures/fig4

The row is a campaign of the agent given the variable descriptions (the gray box), read the
way Figure 3 reads a mathematical one: on the left its key decisions, each a turn of the same
loop -- what it observed (roman), what it inferred (italic), the optimizer it designed (bold)
-- and on the right what those decisions bought, as regret against evaluations. The card panel
is Figure 3's own `cards`, imported rather than copied, so the two figures cannot drift apart.

The point of the row is the middle step, so it is named for where it comes from: the agent
observes scores, but what it infers from them is chemistry -- alkaline-earth nodes bind CO2,
the pore window has a sweet spot -- and that is what picks the next design. Hence the header
reads "Reason from chemistry", and the panel title says so too.

What differs from Figure 3, and why. There the dashed curves are counterfactuals, one per
decision, so each decision owns a colour that ties its card to its curve. Here the other
curves are other methods, not counterfactuals, so a per-decision colour would point at
nothing: the stems take the agent's own colour and the numbered badges tell the decisions
apart. The curve colours are Figure 4a's arm colours instead, so a reader coming from the bar
chart finds the same three things in the same three colours.

The decisions are quoted from the campaign's own decisions.md; the block boundaries printed at
the foot of each card are the real ones (mat_5: blocks after 70 and 85 of 100), and the run
prints the budget and the evaluation at which the pool optimum is reached, against the trace.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .fig3 import BOLD, INK, badge, cards  # noqa: E402
from .figures import STYLE, WIDTH  # noqa: E402

AGENT = "#A878D6"  # the three arm colours of Figure 4a, so the two panels agree
BLACKBOX = "#4C1D6B"
BASELINE = "#7E8A96"
RANDOM = "#C6C6C6"

# method -> (label, colour, linestyle, linewidth), drawn in this order so the agent is on top.
CURVES = [
    ("pool_random", "Random", RANDOM, "-", 1.0),
    ("gp_bo", "GP-BO", BASELINE, (0, (3.2, 1.8)), 1.0),
    ("cheapest_lunch_opus_5_5_high", "Agent, black box", BLACKBOX, "-", 1.0),
    ("cheapest_lunch_opus_5_5_high_domain_context", "Agent, gray box", AGENT, "-", 1.7),
]
GRAY = CURVES[-1][0]

PANELS = {
    "b": dict(
        title="Where the reasoning comes from: MOF chemistry", task_key="mat_5", seed=1,
        unit="mol/kg",
        # The middle step of the loop is named for its source. What the agent infers here is
        # not read off the data -- the data say which picks scored well; the chemistry says
        # why, and so says where to look next.
        steps=("Observe", "Reason from chemistry", "Design"),
        task="Metal-organic frameworks: maximize CO$_2$ uptake at 298 K and 0.15 bar.",
        decisions=[
            dict(after=70, foot="eval 70",
                 observe="Alkaline-earth picks average 4.7 mol/kg; the others 1.5.",
                 reason="Mg/Ca/Sr nodes should hold CO$_2$ by electrostatics.",
                 design="Restrict GP-EI to porous alkaline-earth frameworks.",
                 bought="10 of 15 picks above 6 mol/kg"),
            dict(after=85, foot="eval 85",
                 observe="The GP ranks inside the family poorly; neighbours rank better.",
                 reason="The sweet spot is narrow: pore 4.5\u20136.5 \u00c5, void 0.45\u20130.60.",
                 design="Keep the chemistry rule, swap the ranker for 4 neighbours plus GP.",
                 bought="pool optimum at eval 86"),
        ],
        # Random, GP-BO and the black box run within a third of a decade of each other for
        # most of this campaign, so their labels go past the right edge beside each line's
        # end rather than on top of the curves.
        labels={"Random": (102, 0.46, "left", "center"),
                "GP-BO": (102, -0.18, "left", "center"),
                "Agent, black box": (102, 0.24, "left", "center"),
                "Agent, gray box": (50, -0.95, "center", "center")}),
}


def regret(results: Path, task: str, method: str, seed: int) -> np.ndarray:
    with open(results / "tasks" / task / method / f"seed_{seed}" / "trace.csv") as f:
        return np.array([float(r["simple_regret"]) for r in csv.DictReader(f)])


def curves(results: Path, cfg) -> dict[str, np.ndarray]:
    return {m: regret(results, cfg["task_key"], m, cfg["seed"]) for m, *_ in CURVES}


def panel(ax, cfg, data):
    """Regret against evaluations, log10, with every curve labelled where it runs.

    A campaign that reaches the pool optimum has regret exactly 0, which a log axis has no
    room for. The floor is a decade under the smallest regret anyone reaches and its tick is
    labelled 0, so the drop to the optimum is drawn as what it is rather than hidden.
    """
    nonzero = min(v[v > 0].min() for v in data.values())
    floor = np.log10(nonzero) - 1
    lg = {m: np.log10(np.maximum(v, 10 ** floor)) for m, v in data.items()}
    budget = max(len(v) for v in data.values())

    ax.grid(axis="y", lw=0.35, color="#EBEBEB", zorder=0)
    for i, (m, label, colour, ls, lw) in enumerate(CURVES):
        y = lg[m]
        ax.plot(np.arange(1, len(y) + 1), y, color=colour, lw=lw, ls=ls, zorder=2 + i,
                solid_joinstyle="miter")
        x, ly, ha, va = cfg["labels"][label]
        font = BOLD if m == GRAY else {}
        ax.text(x, ly, label, color="black", fontsize=6.5 + 0.7 * (m == GRAY), ha=ha, va=va,
                clip_on=False, **font)
    for i, d in enumerate(cfg["decisions"]):
        badge(ax, d["after"], lg[GRAY][d["after"] - 1], i + 1, AGENT, size=10)

    top = np.ceil(max(v.max() for v in lg.values()))
    ticks = np.arange(np.ceil(floor + 1), top + 1)
    ax.set_xlim(0, budget)
    ax.set_ylim(floor - 0.22, top + 0.45)
    ax.set_yticks([floor, *ticks])
    ax.set_yticklabels(["0", *[f"$10^{{{int(v)}}}$" for v in ticks]])
    ax.set_xlabel("Evaluations")
    ax.set_ylabel(f"Regret ({cfg['unit']}) " r"$\downarrow$")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)


ROW = 2.85  # inches per campaign row; fig4.py stacks panel a over these


def row(fig, letter, cfg, data, y0, h):
    """One campaign: its decisions on the left, what they bought on the right.

    `y0` and `h` are the row's bottom and height as fractions of the figure, so the same
    layout serves this module's two-row figure and the stacked Figure 4.
    """
    ax_l = fig.add_axes([0.035, y0 + 0.05 * h, 0.44, 0.80 * h])
    ax_r = fig.add_axes([0.585, y0 + 0.17 * h, 0.345, 0.66 * h])
    cards(fig, ax_l, cfg["task"], [{**d, "colour": AGENT} for d in cfg["decisions"]],
          steps=cfg.get("steps", ("Observe", "Reason", "Design")))
    panel(ax_r, cfg, data)
    top = y0 + h - 0.012
    fig.text(0.0, top, letter, fontsize=11, va="top", ha="left", **BOLD)
    fig.text(0.025, top - 0.007, cfg["title"], fontsize=8, va="top", ha="left", color=INK)
    fig.text(ax_r.get_position().x0 - 0.065, top - 0.007, "What those decisions bought",
             fontsize=8, va="top", ha="left", color=INK)
    return ax_l, ax_r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=Path("benchmark/graybox"))
    p.add_argument("--out", type=Path, default=Path("benchmark/figures/fig4"))
    a = p.parse_args(argv)

    plt.rcParams.update(STYLE)
    rows = list(PANELS)
    fig = plt.figure(figsize=(WIDTH, ROW * len(rows)))
    for i, letter in enumerate(rows):
        row(fig, letter, PANELS[letter], curves(a.results, PANELS[letter]),
            1 - (i + 1) / len(rows), 1 / len(rows))

    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"fig4bc.{ext}")
    print(a.out / "fig4bc.png")
    for letter, cfg in PANELS.items():
        data = curves(a.results, cfg)
        g = data[GRAY]
        hit = next((i + 1 for i, v in enumerate(g) if v == 0), None)
        print(f"  {letter}: {cfg['task_key']} seed {cfg['seed']}, budget {len(g)}, "
              f"pool optimum at eval {hit}")


if __name__ == "__main__":
    main()
