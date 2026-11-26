"""Figure 4 of the paper: domain context on real discovery campaigns.

    .venv/bin/python -m aobench.fig4 --out benchmark/figures/fig4

(a) How much faster each arm is than random search at turning up a top-1% candidate, over
    the five scalar tasks of the grey-box campaign, with each arm's geometric mean drawn as a
    rule across the panel. aobench.fig4a.
(b) One of those campaigns in full, in the form of Figure 3: the agent's key decisions on the
    left, and what they bought on the right. Its middle step is named "Reason from chemistry",
    which is the point of the panel -- the agent observes scores, but what it infers from them
    is domain knowledge, and that is what picks the next design. aobench.fig4bc.

The three panels share their colours: whatever is grey in the bar chart is a classical
baseline wherever it appears, vermillion is the agent without the variable descriptions, and
purple is the agent with them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import fig4a, fig4bc  # noqa: E402
from .fig3 import BOLD, INK  # noqa: E402
from .figures import STYLE, WIDTH  # noqa: E402

TITLE_A = "Black-box vs. Grey Box"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=Path("benchmark/graybox"))
    p.add_argument("--pool", type=Path, default=Path("benchmark/graybox/pool"))
    p.add_argument("--branches", type=Path, default=Path("benchmark/graybox/branches"))
    p.add_argument("--out", type=Path, default=Path("benchmark/figures/fig4"))
    a = p.parse_args(argv)

    data, raw = fig4a.collect(a.results, a.pool)
    if not data:
        raise SystemExit(f"no task in {a.results} has all three arms and a reference")
    rows = list(fig4bc.PANELS)
    # Panel a's legend sits above its axes, where the standalone figure has the whole top
    # margin to itself; stacked, it would run into the panel letter. HEAD is the strip of
    # figure given back to it.
    HEAD = 0.26
    h_a = fig4a.SIZE[1]
    height = h_a + HEAD + fig4bc.ROW * len(rows)

    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(WIDTH, height))

    m, (w_a, _) = fig4a.MARGINS, fig4a.SIZE
    y0 = 1 - (h_a + HEAD) / height
    ax_a = fig.add_axes([m["left"] * w_a / WIDTH,
                         y0 + m["bottom"] * h_a / height,
                         (m["right"] - m["left"]) * w_a / WIDTH,
                         (m["top"] - m["bottom"]) * h_a / height])
    fig4a.bars(ax_a, data)
    # Flush with the row below: panel a's ink -- its axis label and tick labels included --
    # starts where panel b's cards start and ends where panel b's plot ends.
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    ink = [ax_a.yaxis.label, *ax_a.get_yticklabels()]
    left_ink = min(inv.transform(a.get_window_extent(r).get_points())[0, 0] for a in ink)
    p_a = ax_a.get_position()
    pad = p_a.x0 - left_ink
    ax_a.set_position([fig4bc.LEFT + pad, p_a.y0, fig4bc.RIGHT - fig4bc.LEFT - pad, p_a.height])
    fig.text(0.0, 1.0, "a", fontsize=11, va="top", ha="left", **BOLD)
    fig.text(0.025, 1 - 0.007 * h_a / height, TITLE_A, fontsize=8, va="top", ha="left", color=INK)

    for i, letter in enumerate(rows):
        cfg = fig4bc.PANELS[letter]
        fig4bc.row(fig, letter, cfg, fig4bc.curves(a.results, cfg),
                   y0 * (len(rows) - 1 - i) / len(rows), y0 / len(rows),
                   fig4bc.branches(a.branches, cfg))

    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"fig4.{ext}", dpi=400)
    print(a.out / "fig4.png")
    print(f"  {'task':8} {'random':>8} {'base':>8} {'black':>8} {'grey':>8}   acceleration")
    for t, r in raw.items():
        print(f"  {t:8} " + "".join(f"{r[x]:8.0f}" for x in ("random", *fig4a.ARMS)) + "   "
              + "  ".join(f"{data[t][x]:.1f}x" for x in fig4a.ARMS))


if __name__ == "__main__":
    main()
