"""Figure 4b as an animation: the grey-box agent closing in on the MOF optimum, beside GP-BO.

    .venv/bin/python -m aobench.fig4_anim --upstream ~/path/to/cheapest_lunch --out fig4b.gif

Two maps of the same landscape, one for each method, and on the right how good each pick is.
The map is pore size (across, log axis) against total void fraction (up), on a 60 x 60 grid:
each cell is coloured by the best MOF in it, gaps filled from their neighbours and the whole
smoothed by a 3 x 3 median, from the pool's median uptake up. A method's picks are not drawn
one by one. Each map carries a single dot, the exponential moving average of the method's picks
in these coordinates (span `SPAN`: the newest pick weighs 2 / (SPAN + 1)), and the line it
leaves behind: where the method has been searching, and where it is heading. The last `RECENT`
steps of the line are drawn bold, the older ones faint, so the current heading stands out.
Around the dot, an ellipse one standard deviation wide: the moving covariance of the picks,
with the same weights. An average alone hides how scattered the picks behind it are -- the
mean of picks spread over the whole map lands in its middle, near the optimum or not -- and
the ellipse shows it.

The agent's map follows its own search space. At evaluation 55 it bounded pore size to
3.8-8.5 A and void fraction to 0.20-0.65 (with density and accessibility; 8,540 -> 2,782 MOFs,
every one inside those bounds). At decision 1 it kept only alkaline-earth frameworks (118 MOFs,
21 of the pool's 86 top-1% and the optimum), and at decision 2 only its sweet spot within them
(39 MOFs, `SWEET`). The landscape is never recoloured: at each restriction the agent's map dims
every cell holding none of the MOFs still considered, and the box shrinks to the smallest one
around the cells left. Its map zooms once, at evaluation 55; GP-BO's map keeps the whole pool in
view, with the agent's box outlined on it. The first two masks are not redrawn by hand: they
are the `allowed` pools of the agent's own block scripts, constructed exactly as the campaign
constructed them (see `aobench.graybox_branches`). The third is the region decisions.md states, and every pick after
it lies inside. GP-BO searches all 8,540 throughout; its map is never dimmed.

On the right, each pick's rank in the pool against evaluations, with the same moving average
(of the log rank): the agent's climbs from the middle of the pool to its top 1% as its search
space narrows. The walk pauses at each key decision with its reasoning across the top, in
three short lines (`MOMENTS`: what it saw, what it concluded, what it did), and the
star flashes when the optimum is found.

The pool's features and the block scripts are not in this repository, so this needs the
collaborators' repository; the campaign's picks are read from this one. A `.mp4` output is
written with ffmpeg, anything else as a GIF.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402
from scipy import ndimage as nd  # noqa: E402

from . import fig4bc  # noqa: E402
from .fig2 import SCALE  # noqa: E402
from .figures import STYLE  # noqa: E402

COLOURS = ["#D9D9D9", "#D55E00", "#0072B2"]  # before decision 1, after 1, after 2
GREY = "#BFC5CC"  # GP-BO's path
N_GRID = 60
SPAN = 8        # the moving average's span
RECENT = 20     # the last RECENT steps of a path drawn bold; older ones faint
DI_MIN, DI_MAX = 3.2, 25.0  # pore sizes below DI_MIN (a handful of zeros) are clipped for the log axis
VOID_MAX = 0.9  # top of the map (0.1% of the pool lies above)
FLOOR = 1.4     # the colour scale starts at the pool's median uptake (mol/kg)
DIM = 0.22      # opacity of the landscape outside the agent's search space
GOLD = "#FFD54A"
# the blocks whose pool restriction the agent's map shows, with what each kept
RESTRICTIONS = [("004_gpei3", "porous frameworks of plausible geometry"),
                ("005_gpei4", "porous alkaline-earth frameworks")]
BOX = (3.8, 8.5, 0.20, 0.65)   # block 004's bounds: pore size (A), total void fraction
ZOOM = (3.5, 9.8, 0.12, 0.72)  # the agent's view from evaluation 55: the box, and its path after it
# The three moments the walk stops at, keyed by evaluation, condensed from the campaign's own
# decisions.md (blocks 4, 5 and 6): what it saw, what it concluded, what it did.
MOMENTS = {
    55: ("Evaluation 55", [
        ("Observe", "Top picks all have small pores"),
        ("Reason", "CO$_2$ binds best in narrow pores"),
        ("Design", "Search small pores only")]),
    70: ("Evaluation 70  \u00b7  Decision 1", [
        ("Observe", "Mg/Ca/Sr picks score 3\u00d7 higher"),
        ("Reason", "Alkaline-earth metals bind CO$_2$"),
        ("Design", "Search alkaline-earth MOFs only")]),
    85: ("Evaluation 85  \u00b7  Decision 2", [
        ("Observe", "The model ranks these poorly"),
        ("Reason", "The best hits share one geometry"),
        ("Design", "Search that sweet spot only")]),
}


def _gapfill(g):
    return np.where(g > 0, g, nd.maximum_filter(g, size=3))


SMOOTHERS = {"none": lambda g: g, "fill": _gapfill, "max": lambda g: nd.maximum_filter(g, size=3),
             "median": lambda g: nd.median_filter(_gapfill(g), size=3)}


def pool(upstream: Path, task: str):
    """(pore size, total void fraction) and uptake for every MOF in the task table."""
    d = upstream / "data" / task
    names = next(csv.reader(open(d / "meta.csv")))[1:]
    with open(d / "main.csv") as f:
        head = next(csv.reader(f))
    col = {names[int(c[1:]) - 1]: i for i, c in enumerate(head) if c.startswith("x")}
    use = [col["Di"], col["POAVF"], col["PONAVF"], head.index("y")]
    a = np.loadtxt(d / "main.csv", delimiter=",", skiprows=1, usecols=use)
    return np.c_[a[:, 0], a[:, 1] + a[:, 2]], a[:, 3]


def grid(P, y, ext, nx, ny, smooth="median"):
    """The best y in each cell of an nx x ny grid over ext, smoothed."""
    ix = ((P[:, 0] - ext[0]) / (ext[1] - ext[0]) * nx).astype(int)
    iy = ((P[:, 1] - ext[2]) / (ext[3] - ext[2]) * ny).astype(int)
    k = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    g = np.zeros((ny, nx))
    np.maximum.at(g, (iy[k], ix[k]), y[k])
    return SMOOTHERS[smooth](g)


def landscape(P, y, ext, smooth="median", n=N_GRID):
    return grid(P, y, ext, n, n, smooth)


def ema(Q, ids, span=SPAN):
    """The exponential moving average of the picks' coordinates, after each evaluation."""
    a, out = 2 / (span + 1), [np.asarray(Q[ids[0]], float)]
    for i in ids[1:]:
        out.append((1 - a) * out[-1] + a * Q[i])
    return np.array(out)


def spread(Q, ids, span=SPAN):
    """The moving average of the picks and their moving covariance, after each evaluation.

    The same weights as `ema`: each new pick pulls the mean 2 / (span + 1) of the way to it,
    and the covariance is the exponentially weighted one about that mean.
    """
    a = 2 / (span + 1)
    m, C = np.asarray(Q[ids[0]], float), np.zeros((Q.shape[1], Q.shape[1]))
    ms, Cs = [m], [C]
    for i in ids[1:]:
        d = Q[i] - m
        m = m + a * d
        C = (1 - a) * (C + a * np.outer(d, d))
        ms.append(m)
        Cs.append(C)
    return np.array(ms), np.array(Cs)


def ellipse(C, k=1.0):
    """Width, height and angle (degrees) of the k-s.d. ellipse of covariance C."""
    w, v = np.linalg.eigh(C)
    w = np.sqrt(np.clip(w, 0, None))
    return 2 * k * w[1], 2 * k * w[0], np.degrees(np.arctan2(v[1, 1], v[0, 1]))


def picks(results: Path, task: str, method: str, seed: int) -> np.ndarray:
    with open(results / "tasks" / task / method / f"seed_{seed}" / "trace.csv") as f:
        return np.array([int(r["candidate_index"]) for r in csv.DictReader(f)])


def search_spaces(upstream: Path, task: str, seed: int):
    """[(evaluations done, mask, what it kept)]: the pool each restricting block allowed.

    Each block is constructed as the campaign constructed it, and its `allowed` read off.
    """
    from . import graybox_branches as G
    run, space, *_ = G.campaign(upstream, task, seed)
    out = []
    for block, what in RESTRICTIONS:
        meta = json.loads((run / f"agent_artifacts/blocks/{block}.json").read_text())
        opt = G.load(run / f"agent_artifacts/blocks/{block}.py").Optimizer(space, **meta["args"])
        out.append((meta["after"], np.asarray(opt.allowed, bool), what))
    return out


# Decision 2's sweet spot, in the ranges decisions.md gives for it (block 6): inside block 5's
# alkaline-earth pool, pore size, accessible void fraction and density bounded, surface nonzero.
# Block 6 is a fixed list, so it has no `allowed` to read; every pick it and block 7 made lies
# inside this region, which `build` checks.
SWEET = {"Di": (4.2, 7.0), "POAVF": (0.36, 0.66), "density [g/cm^3]": (0.7, 1.1)}


def sweet_spot(upstream: Path, task: str, within: np.ndarray) -> np.ndarray:
    d = upstream / "data" / task
    names = next(csv.reader(open(d / "meta.csv")))[1:]
    with open(d / "main.csv") as f:
        head = next(csv.reader(f))
    col = {names[int(c[1:]) - 1]: i for i, c in enumerate(head) if c.startswith("x")}
    keys = [*SWEET, "ASA [m^2/cm^3]"]
    X = np.loadtxt(d / "main.csv", delimiter=",", skiprows=1, usecols=[col[k] for k in keys])
    m = within & (X[:, -1] > 0)
    for j, (lo, hi) in enumerate(SWEET.values()):
        m &= (X[:, j] >= lo) & (X[:, j] <= hi)
    return m


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


HALO = [pe.Stroke(linewidth=3.6, foreground="#111111"), pe.Normal()]


def layout(height=4.8):
    """The figure the animations share: two maps, their colour bar, the climb and a caption.

    The plots keep their size in inches whatever the height; a taller figure gives the caption
    across the top more room.
    """
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(13, height), dpi=110)
    b, h = 0.528 / height, 2.664 / height
    axp = fig.add_axes([0.06, b, 0.25, h])
    axg = fig.add_axes([0.33, b, 0.25, h])
    cax = fig.add_axes([0.59, b, 0.007, h])
    axr = fig.add_axes([0.668, b, 0.317, h])
    cap = fig.text(0.02, 0.975, "", va="top", ha="left", fontsize=9.5, linespacing=1.45)
    for ax, title in ((axp, "Agent, grey box"), (axg, "GP-BO (best baseline)")):
        ax.set_facecolor("#0E0A1C")
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold", pad=16)
    info = {ax: ax.text(1.0, 1.015, "", transform=ax.transAxes, va="bottom", ha="right",
                        fontsize=6.8, linespacing=1.35) for ax in (axp, axg)}
    return fig, (axp, axg, cax, axr), cap, info


class Walker:
    """A method's moving average on a map: one dot, the spread ellipse, and the dot's path.

    `colours[i]` colours the dot and the step it takes at pick i. The last `RECENT` steps of
    the path are bold, the older ones faint.
    """

    def __init__(self, ax, Q, ids, colours):
        self.E, self.C = spread(Q, ids)
        self.cs = list(colours)
        self.cloud = ax.add_patch(Ellipse((0, 0), 0, 0, lw=1.0, ec=(1, 1, 1, 0.75), zorder=4.5))
        self.faint = ax.add_collection(LineCollection([], linewidths=1.1, capstyle="butt", zorder=4))
        self.bold = ax.add_collection(LineCollection([], linewidths=2.0, capstyle="round",
                                                     joinstyle="round", zorder=5, path_effects=HALO))
        self.dot, = ax.plot([], [], "o", ms=11, mec="white", mew=1.6, zorder=6)

    def at(self, e):
        """Show the walk after e evaluations (hidden at 0)."""
        self.cloud.set_visible(e > 0)
        self.dot.set_visible(e > 0)
        if e == 0:
            self.bold.set_segments([])
            self.faint.set_segments([])
            return
        E, cs = self.E, self.cs[:e]
        wd, ht, ang = ellipse(self.C[e - 1])
        self.cloud.set_center(E[e - 1])
        self.cloud.set_width(wd)
        self.cloud.set_height(ht)
        self.cloud.set_angle(ang)
        self.cloud.set_facecolor(to_rgba(cs[-1], 0.22))
        seg = np.stack([E[:e - 1], E[1:e]], 1) if e > 1 else np.empty((0, 2, 2))
        old = np.arange(e - 1) < e - 1 - RECENT  # steps more than RECENT evaluations back
        self.bold.set_segments(seg[~old])
        self.bold.set_color([c for c, o in zip(cs[1:], old) if not o] or "none")
        self.faint.set_segments(seg[old])
        self.faint.set_color([to_rgba(c, 0.4) for c, o in zip(cs[1:], old) if o] or "none")
        self.dot.set_data([E[e - 1, 0]], [E[e - 1, 1]])
        self.dot.set_markerfacecolor(cs[-1])

    def distance(self, target, sd):
        """How far the dot is from `target` after each evaluation, in units of sd per axis."""
        return np.linalg.norm((self.E - target) / sd, axis=1)

    def width(self, sd):
        """How wide the picks are after each evaluation: the geometric-mean s.d., in sd units."""
        return np.clip(np.linalg.det(self.C / np.outer(sd, sd)), 0, None) ** 0.25


def rank(v, ids):
    """Each pick's rank in the pool as 'top x %' (higher v is better): the share as good."""
    return 100 * (v[None, :] >= v[ids][:, None]).mean(1)


def climb(axr, ra, rg, N, colours, marks, found, gfound=None, xlabel="Evaluations",
          legend_x=0.0):
    """Each pick's rank against evaluations, with its moving average (of the log rank).

    The agent's picks are dotted in `colours`, GP-BO's grey; `marks` puts a labelled dotted
    line at each of its evaluations once reached, and a star marks where each method reaches
    the optimum. The legend sits in the top left, `legend_x` of the way across (move it off
    an early star). Returns update(e).
    """
    n = len(ra)
    smooth = lambda r: 10 ** ema(np.log10(r)[:, None], np.arange(len(r)))[:, 0]  # noqa: E731
    ma, mg = smooth(ra), smooth(rg)
    axr.set_yscale("log")
    axr.set_ylim(130, 100 / N / 1.9)
    axr.set_xlim(0, n + 1)
    axr.set_yticks([100, 10, 1, 0.1])
    axr.set_yticklabels(["100", "10", "1", "0.1"])
    axr.set_ylabel("Top k% candidate")
    axr.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axr.tick_params(axis="y", length=0)
    axr.set_xlabel(xlabel)
    axr.set_title("How good each pick is", fontsize=9, loc="left",
                  fontweight="bold", pad=16)
    axr.grid(axis="y", lw=0.35, color="#EBEBEB")
    for sp in ("top", "right", "left"):
        axr.spines[sp].set_visible(False)
    gdots = axr.scatter([], [], s=9, c="#C9CED3", edgecolors="none", zorder=2)
    adots = axr.scatter([], [], s=15, edgecolors="black", linewidths=0.3, zorder=3)
    gmed, = axr.plot([], [], color=fig4bc.BASELINE, lw=1.6, zorder=4)
    amed, = axr.plot([], [], color=fig4bc.AGENT, lw=2.2, zorder=5)
    hit, = axr.plot([], [], marker="*", ms=13, mfc=GOLD, mec="black", mew=0.6, ls="", zorder=6)
    ghit, = axr.plot([], [], marker="*", ms=11, mfc="#C9CED3", mec="black", mew=0.6, ls="",
                     zorder=6)
    axr.legend([Line2D([], [], color=fig4bc.AGENT, lw=2.2), Line2D([], [], color=fig4bc.BASELINE, lw=1.6)],
               ["Agent, grey box", "GP-BO"], loc="upper left", bbox_to_anchor=(legend_x, 1.0),
               fontsize=6.8, frameon=False)
    at = sorted(marks)
    vls = [axr.axvline(c, color="#9A9A9A", lw=0.7, ls=":", zorder=1, visible=False) for c in at]
    vtx = [axr.text(c - 0.6 * n / 100, 100 / N * 1.5, marks[c], fontsize=6.3, color="#666",
                    rotation=90, va="top", ha="right", visible=False) for c in at]

    def update(e):
        ev = np.arange(1, e + 1)
        adots.set_offsets(np.c_[ev, ra[:e]])
        adots.set_facecolor(list(colours[:e]) or "none")
        gdots.set_offsets(np.c_[ev, rg[:e]])
        amed.set_data(ev, ma[:e])
        gmed.set_data(ev, mg[:e])
        hit.set_data(([found], [100 / N]) if e >= found else ([], []))
        ghit.set_data(([gfound], [100 / N]) if gfound and e >= gfound else ([], []))
        for c, ln, tx in zip(at, vls, vtx):
            ln.set_visible(e >= c)
            tx.set_visible(e >= c)
    return update


def save(fig, draw, frames, out):
    """Write the animation: ffmpeg for a .mp4, otherwise a GIF."""
    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=10, bitrate=4000) if out.suffix == ".mp4" else PillowWriter(fps=10)
    anim.save(out, writer=writer, dpi=150)


def build(a):
    """The storyboard, the figure and its draw function."""
    up = a.upstream.expanduser()
    cfg = fig4bc.PANELS["b"]
    task, seed, method = cfg["task_key"], cfg["seed"], fig4bc.GRAY
    P, y = pool(up, task)
    spaces = search_spaces(up, task, seed)
    spaces.append((fig4bc.PANELS["b"]["decisions"][1]["after"],
                   sweet_spot(up, task, spaces[-1][1]), "its sweet spot"))
    idx = picks(a.results, task, method, seed)
    gidx = picks(a.results, task, "gp_bo", seed)
    decisions = cfg["decisions"]
    cuts = [d["after"] for d in decisions]
    n, N = len(idx), len(y)
    best = int(np.argmax(y))
    found = next(i + 1 for i, v in enumerate(idx) if v == best)
    gfound = next((i + 1 for i, v in enumerate(gidx) if v == best), None)
    m55, m70, m85 = (mk for _a, mk, _w in spaces)
    if not m85[idx[spaces[2][0]:]].all():
        raise SystemExit("a pick after decision 2 lies outside the sweet spot drawn for it")
    if not np.all((P[m55, 0] >= BOX[0]) & (P[m55, 0] <= BOX[1]) & (P[m55, 1] >= BOX[2])
                  & (P[m55, 1] <= BOX[3])):
        raise SystemExit("the agent's first search space does not lie inside the box drawn for it")

    # ---- the storyboard: one dict per frame -------------------------------------------
    # e: evaluations shown; stage: how far the agent's search space has narrowed (0 = the
    # whole pool, k = after the k-th restriction, fractions mid-transition); dec: a decision
    # on screen; flash: frames since the optimum was found.
    narrow_at = {after: k for k, (after, *_r) in enumerate(spaces)}
    frames, stage = [], 0.0

    def emit(e, **kw):
        frames.append({"e": e, "stage": stage, "dec": None, "flash": None, "note": None,
                       "end": False, **kw})

    HOLD, STEPS = 18, (20, 12, 12)  # frames held on a caption; frames to zoom, then to narrow again
    for e in range(1, n + 1):
        emit(e)
        tag = {"dec": cuts.index(e)} if e in cuts else {"note": narrow_at.get(e)}
        if e in cuts or e in narrow_at:
            for _ in range(HOLD):
                emit(e, **tag)
            if e in narrow_at:
                k = narrow_at[e]
                for t in range(1, STEPS[k] + 1):
                    emit(e, stage=k + t / STEPS[k], **tag)
                stage = k + 1.0
            for _ in range(HOLD):
                emit(e, **tag)
        if e == found:
            for t in range(24):
                emit(e, flash=t)
    for _ in range(36):
        emit(n, end=True)

    # ---- geometry -----------------------------------------------------------------------
    lg = np.log10
    Q = np.c_[lg(np.clip(P[:, 0], DI_MIN, None)), P[:, 1]]  # plot coordinates
    V0 = (lg(DI_MIN), lg(DI_MAX), 0.0, VOID_MAX)
    V1 = (lg(ZOOM[0]), lg(ZOOM[1]), ZOOM[2], ZOOM[3])

    def view(s):
        t = smoothstep(s)
        return tuple((1 - t) * u + t * w for u, w in zip(V0, V1))

    # ---- the figure ---------------------------------------------------------------------
    fig, (axp, axg, cax, axr), _cap, info = layout(height=5.25)
    cmap = SCALE.reversed()
    kw = dict(origin="lower", aspect="auto", cmap=cmap, vmin=FLOOR, vmax=y.max(),
              interpolation="nearest")
    base = {}
    for ax in (axp, axg):
        base[ax] = ax.imshow(landscape(Q, y, V0), extent=V0, zorder=0, **kw)
        ax.set_xlabel("Pore size, Di (Å)")
    axp.set_ylabel("Total void fraction")
    # The agent's search space never recolours the landscape: a cell with none of the MOFs the
    # agent still considers is dimmed, and the box is the smallest one around the cells left.
    # Stage 0 is the whole pool; stage k, the k-th restriction.
    dx, dy = (V0[1] - V0[0]) / N_GRID, (V0[3] - V0[2]) / N_GRID
    occ = [np.ones((N_GRID, N_GRID), bool)] + [
        landscape(Q[mk], np.ones(int(mk.sum())), V0, "none") > 0 for _a, mk, _w in spaces]

    def bounds(o):
        cols, rows = np.flatnonzero(o.any(0)), np.flatnonzero(o.any(1))
        return np.array([V0[0] + cols[0] * dx, V0[0] + (cols[-1] + 1) * dx,
                         V0[2] + rows[0] * dy, V0[2] + (rows[-1] + 1) * dy])
    boxes = [np.array(V0)] + [bounds(o) for o in occ[1:]]
    shade = np.zeros((N_GRID, N_GRID, 4))
    shade[..., :3] = to_rgba(axp.get_facecolor())[:3]
    dim = axp.imshow(shade, extent=V0, origin="lower", aspect="auto", interpolation="nearest",
                     zorder=1)
    box = axp.add_patch(Rectangle((0, 0), 0, 0, fill=False, ec="white", lw=1.2, zorder=3,
                                  alpha=0.0))
    # on GP-BO's map, the same box: where the agent is searching
    window = axg.add_patch(Rectangle((0, 0), 0, 0, fill=False, ec="white", lw=1.3,
                                     ls=(0, (4, 2.5)), zorder=3, alpha=0.0))
    wlabel = axg.text(0, 0, "agent's search space", color="white", fontsize=8, va="bottom",
                      ha="left", zorder=3, alpha=0.0)

    def stage_at(s):
        """The dimming (0 kept, 1 dimmed) and the box at stage s, eased between stages."""
        k = min(int(s), len(spaces) - 1)
        t = smoothstep(s - k)
        d = (1 - t) * ~occ[k] + t * ~occ[k + 1]
        return d, (1 - t) * boxes[k] + t * boxes[k + 1]
    stars = [ax.plot(*Q[best], marker="*", ms=15, mfc="white", mec="black", mew=0.6, zorder=8)[0]
             for ax in (axp, axg)]
    pulse, = axp.plot([], [], "o", mfc="none", mec=GOLD, mew=2.0, zorder=7)
    cb = fig.colorbar(base[axp], cax=cax, extend="min")
    cb.set_label("Best uptake in the cell (mol/kg)", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    phase = lambda i: sum(i >= c for c in cuts)  # noqa: E731  evaluation i (0-based) -> colour
    colours = [COLOURS[phase(i)] for i in range(n)]
    walk = [Walker(axp, Q, idx, colours), Walker(axg, Q, gidx, [GREY] * n)]
    inside = lambda E: (E[:, 0] >= V1[0]) & (E[:, 0] <= V1[1]) & (E[:, 1] >= V1[2]) & (E[:, 1] <= V1[3])  # noqa: E731
    z = spaces[0][0]
    if not inside(walk[0].E[z - 1:]).all():
        raise SystemExit("the agent's path leaves its zoomed view after the zoom; widen ZOOM")
    sd = Q.std(0)
    da, dg = (w.distance(Q[best], sd) for w in walk)
    wa, wg = (w.width(sd) for w in walk)

    def ticks(ax, v):
        xt = [t for t in (3.5, 4, 5, 6, 7, 8, 10, 15, 20) if v[0] <= lg(t) <= v[1]]
        ax.set_xticks(lg(xt))
        ax.set_xticklabels([f"{t:g}" for t in xt])
        vt = np.arange(0, 1.0, 0.1 if v[3] - v[2] < 0.7 else 0.2)
        ax.set_yticks([t for t in vt if v[2] <= t <= v[3]])
        ax.set_yticklabels([f"{t:.1f}" for t in vt if v[2] <= t <= v[3]] if ax is axp else [])

    ra, rg = rank(y, idx), rank(y, gidx)
    climbed = climb(axr, ra, rg, N, colours, {spaces[0][0]: "pore box", cuts[0]: "decision 1",
                                              cuts[1]: "decision 2"}, found, gfound)

    late_a, late_g = np.median(ra[cuts[0]:]), np.median(rg[cuts[0]:])

    INK = "#222222"
    TOP, STEP, LX, BX = 0.972, 0.073, 0.02, 0.09
    CAP = dict(va="top", ha="left", family="STIXGeneral")  # STIX Two Text has no bold here
    head = fig.text(LX, TOP, "", fontsize=15, fontweight="bold", **CAP)
    labels = [fig.text(LX, TOP - (i + 1) * STEP - 0.012, "", fontsize=12, fontweight="bold",
                       **CAP) for i in range(3)]
    bodies = [fig.text(BX, TOP - (i + 1) * STEP - 0.008, "", fontsize=15, color=INK,
                       **CAP) for i in range(3)]

    def caption(title, rows=(), colour=INK):
        """The heading in `colour`, then rows of (label, text); an empty label spans the row."""
        head.set_text(title)
        head.set_color(colour)
        for i, (lt, bt) in enumerate(zip(labels, bodies)):
            lab, body = rows[i] if i < len(rows) else ("", "")
            lt.set_text(lab.upper())
            lt.set_color(colour)
            bt.set_text(body)
            bt.set_x(BX if lab else LX)
            bt.set_fontweight("bold" if lab == "Design" else "normal")

    colour_at = {spaces[0][0]: INK, cuts[0]: COLOURS[1], cuts[1]: COLOURS[2]}

    def draw(fi):
        f = frames[fi]
        e, s = f["e"], f["stage"]
        s1 = np.clip(s, 0, 1)
        v = view(s1)
        for ax, vv in ((axp, v), (axg, V0)):
            ax.set_xlim(vv[:2])
            ax.set_ylim(vv[2:])
            ticks(ax, vv)
        d, bx = stage_at(s)
        shade[..., 3] = (1 - DIM) * d
        dim.set_data(shade)
        for r in (box, window):
            r.set_xy((bx[0], bx[2]))
            r.set_width(bx[1] - bx[0])
            r.set_height(bx[3] - bx[2])
            r.set_alpha(float(0.9 * s1))
        wlabel.set_position((bx[0] + 0.01, bx[3] + 0.012))
        wlabel.set_alpha(float(0.9 * s1))

        for w in walk:
            w.at(e)
        got = e >= found
        stars[0].set_markerfacecolor(GOLD if got else "white")
        stars[0].set_markersize(19 if got else 15)
        if f["flash"] is not None:
            t = (f["flash"] % 12) / 12
            pulse.set_data([Q[best, 0]], [Q[best, 1]])
            pulse.set_markersize(14 + 30 * t)
            pulse.set_alpha(1 - t)
        else:
            pulse.set_data([], [])

        climbed(e)

        # the caption across the top
        di, nk = f["dec"], f["note"]
        if f["end"]:
            caption("")
        elif f["flash"] is not None:
            caption(f"Evaluation {found}  \u00b7  Optimum found", colour=COLOURS[2])
        elif di is not None or nk is not None:
            at = decisions[di]["after"] if di is not None else spaces[nk][0]
            caption(*MOMENTS[at], colour_at[at])
        elif e < spaces[0][0]:
            caption(f"Maximize CO$_2$ uptake across {N:,} MOFs")
        else:
            k = max(c for c in colour_at if c <= e)
            caption(MOMENTS[k][0], MOMENTS[k][1][2:], colour_at[k])
        return ()

    last = slice(n - 30, n)
    summary = dict(frames=len(frames), optimum=round(float(y.max()), 2), agent=found, gpbo=gfound,
                   spaces=[(after, int(mk.sum()), bool(mk[best])) for after, mk, _w in spaces],
                   late=(round(float(late_a), 2), round(float(late_g), 1)),
                   # how far each average sits from the optimum over the last 30, in pool s.d.
                   dist_last30=(round(float(da[last].mean()), 2), round(float(dg[last].mean()), 2)),
                   spread_last30=(round(float(wa[last].mean()), 2), round(float(wg[last].mean()), 2)))
    return fig, draw, frames, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--results", type=Path, default=Path("benchmark/graybox"))
    p.add_argument("--out", type=Path, default=Path("benchmark/figures/fig4/fig4b.gif"))
    a = p.parse_args(argv)
    fig, draw, frames, summary = build(a)
    save(fig, draw, frames, a.out)
    print(a.out, summary)


if __name__ == "__main__":
    main()
