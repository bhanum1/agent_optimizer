"""The mixture task as an animation, in the form of `aobench.fig4_anim`.

    .venv/bin/python -m aobench.mix_anim --upstream ~/path/to/cheapest_lunch --out mix.gif

mix_1: binary mixtures, minimize viscosity at room temperature. Two maps of the same landscape
-- mole-fraction-weighted molecular weight against mole-fraction-weighted H-bond donor count,
each cell of a 60 x 60 grid coloured by its lowest-viscosity mixture (log colour, floor at the
pool median), gaps filled and 3 x 3 median-smoothed -- one for each method, and on the right how
good each pick is. Of a dozen descriptor pairs these two predict log viscosity best (R^2 0.66
from the two alone); the mole-fraction weighting is the log-linear mixing rule. Mixtures beyond
the axes (diol- and water-rich ones above 1.15 donors, a few over 250 g/mol) are drawn on the
edge rather than dropped. Two-donor-free or two-alcohol pairs sit on the lines at 0 and 1
donors whatever their fractions: that is the descriptor, not the plot.

As in Figure 4b's animation, the picks are not drawn one by one: each map carries a single dot,
the moving average of the method's picks, the ellipse of their moving spread, and the path the
dot leaves (`aobench.fig4_anim.Walker`); the right panel is the same rank-in-the-pool climb. The
map is not zoomed: both paths and their ellipses stay well inside the whole map.

`--seed` picks the campaign (default 1; 0 is the other one written up in `CAMPAIGNS`). The
default is grey-box seed 1, against GP-BO seed 1, over the first `END` of 250 evaluations.
It is the agent's middle seed (its seeds reached the optimum at 8, 44 and 48; GP-BO's at 44-159,
159 on this one), chosen over seed 2, which reached it at evaluation 8, during its opening
screen, leaving no path to show. Seed 1's three decisions in the window, from its decisions.md:
before any measurement, 14 mixtures its chemistry prior ranks lowest plus 10 random ones to
test the prior; at 24, a latent pure-component model with literature viscosities as its prior,
searched by expected improvement; at 54, the same model with a negative excess term for
alcohol + aprotic pairs. None restricts the pool, so neither map is dimmed. The block
boundaries are read from the agent's own block files, and the 14 / 10 split of the first block
must reproduce the value ranges its decisions.md reports for the two groups.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from .fig2 import SCALE  # noqa: E402
from .fig4_anim import (COLOURS, GOLD, GREY, Walker, climb, landscape, layout, picks,  # noqa: E402
                        rank, save)

TASK, METHOD = "mix_1", "cheapest_lunch_opus_5_5_high_domain_context"
N_GRID = 60
EXT = (15.0, 250.0, -0.04, 1.16)  # weighted molecular weight (g/mol), weighted H-bond donors
END = 60                          # evaluations shown, of the 250
RANDOM = "#FFFFFF"
# the agent's colour after each decision: block 1, then the model, then the model revised
AGENT = [COLOURS[1], COLOURS[2], "#009E73"]
# Each campaign's decisions in the window, from its decisions.md, with the block each began.
# `split`: block 1 is the prior's picks first and then random ones, and the two groups' value
# ranges (cP), as decisions.md reports them, which the run checks.
CAMPAIGNS = {
    1: dict(
        split=(14, ((0.51, 1.06), (0.71, 13.6))),
        decisions=[
            dict(after=0, block="001_init_design",
                 reason="viscosity rises with molecular weight, H-bond donors (alcohols, glycols, "
                        "water), polarity and rings; mixtures mix log-linearly.",
                 design="measure the 14 mixtures this prior ranks lowest, plus 10 random ones to "
                        "test it against."),
            dict(after=24, block="002_latent_ei",
                 observe="the prior's picks measure {prior} cP; the random ones {random} cP.",
                 reason="a mixture's log viscosity is its components' pure ones, mixed by mole "
                        "fraction, and the literature knows most pure values.",
                 design="a latent pure-component model with literature viscosities as its prior, "
                        "searched by expected improvement over the whole pool."),
            dict(after=54, block="003_latent_ei2",
                 observe="the model's residuals run negative where an alcohol meets an aprotic "
                         "solvent.",
                 reason="H-bond-asymmetric pairs mix below the log-linear rule (a negative "
                        "Grunberg–Nissan excess).",
                 design="the same model with an excess term for those pairs; still expected "
                        "improvement."),
        ],
        after=["Block 1: the prior's 14 lowest (orange on the right), then 10 random mixtures "
               "(white).",
               "After decision 2: a latent pure-component model, literature viscosities as its "
               "prior, searches the whole pool by expected improvement.",
               "After decision 3: the same model, with an excess term for alcohol + aprotic "
               "pairs."]),
    0: dict(
        split=None,
        decisions=[
            dict(after=0, block="001_prior_probe",
                 reason="a low-viscosity solvent (MTBE, acetonitrile, alkanes, acetone, esters, "
                        "ethers) at high fraction, paired with an alcohol, mixed log-ideally.",
                 design="measure 24 distinct pairs this prior ranks lowest, covering about 20 "
                        "solvent families."),
            dict(after=24, block="002_structbo",
                 observe="the prior's picks measure {block1} cP: the right level, but it ranks "
                         "them poorly.",
                 reason="log-ideal mixing of pure viscosities, plus a pair excess; pairs that share "
                        "a molecule share what is learned about it.",
                 design="a structured Bayesian mixture model of that form, searched by expected "
                        "improvement over the whole pool."),
            dict(after=44, block="003_structbo2",
                 observe="20 model picks, no improvement; 4 went to near-identical C8 aromatics.",
                 reason="chemically similar molecules have similar viscosities.",
                 design="the same model, sharing molecule effects through a descriptor-similarity "
                        "kernel; still expected improvement."),
        ],
        after=["Block 1: the 24 pairs the chemistry prior ranks lowest.",
               "After decision 2: a structured mixture model searches the whole pool by expected "
               "improvement.",
               "After decision 3: the same model, with molecules sharing effects by similarity."]),
}


def pool(upstream: Path):
    """Coordinates, viscosity, fractions and component SMILES for every mixture."""
    d = upstream / "data" / TASK
    with open(d / "meta.csv") as f:
        head = next(csv.reader(f))
    want = ["mole_fraction_1", "mole_fraction_2", "label"] + [
        f"component_{k}_{n}" for k in (1, 2) for n in ("MolWt", "NumHDonors")]
    a = np.loadtxt(d / "meta.csv", delimiter=",", skiprows=1, usecols=[head.index(c) for c in want])
    col = dict(zip(want, a.T))
    x = np.c_[col["mole_fraction_1"], col["mole_fraction_2"]]
    with open(d / "id.csv") as f:
        smiles = [json.loads(r["smiles"]) for r in csv.DictReader(f)]
    w = lambda n: x[:, 0] * col[f"component_1_{n}"] + x[:, 1] * col[f"component_2_{n}"]  # noqa: E731
    return np.c_[w("MolWt"), w("NumHDonors")], col["label"], x, smiles


def first_hits(results: Path, method: str) -> list[int]:
    """The evaluation at which each seed of a method first reached the pool optimum."""
    out = []
    for d in sorted((results / "tasks" / TASK / method).glob("seed_*")):
        with open(d / "trace.csv") as f:
            r = [float(v["simple_regret"]) for v in csv.DictReader(f)]
        out.append(next((i + 1 for i, v in enumerate(r) if v == 0), None))
    return out


def build(a):
    """The storyboard, the figure and its draw function."""
    up = a.upstream.expanduser()
    P, eta, x, smiles = pool(up)
    cfg, SEED = CAMPAIGNS[a.seed], a.seed
    DECISIONS = cfg["decisions"]
    full, gfull = picks(a.results, TASK, METHOD, SEED), picks(a.results, TASK, "gp_bo", SEED)
    idx, gidx = full[:END], gfull[:END]
    n, N = len(idx), len(eta)
    best = int(np.argmin(eta))
    found = next((i + 1 for i, v in enumerate(idx) if v == best), None)
    gfound = next((i + 1 for i, v in enumerate(gidx) if v == best), None)
    gever = next((i + 1 for i, v in enumerate(gfull) if v == best), None)
    if found is None:
        raise SystemExit(f"the agent does not reach the optimum in the first {END} evaluations")
    # the decisions are where the agent's blocks begin, and block 1 splits as it says it does
    blocks = up / "results/tasks" / TASK / METHOD / f"seed_{SEED}" / "agent_artifacts/blocks"
    for d in DECISIONS:
        if json.loads((blocks / f"{d['block']}.json").read_text())["after"] != d["after"]:
            raise SystemExit(f"block {d['block']} does not start after {d['after']} evaluations")
    cuts = [d["after"] for d in DECISIONS[1:]]
    span = lambda k: f"{eta[k].min():.2f}–{eta[k].max():.2f}"  # noqa: E731
    fill = dict(block1=span(idx[:cuts[0]]))
    n_prior = cuts[0]
    if cfg["split"]:
        n_prior, reported = cfg["split"]
        groups = (idx[:n_prior], idx[n_prior:cuts[0]])
        for g, (lo, hi) in zip(groups, reported):
            if not (np.isclose(eta[g].min(), lo, atol=0.006)
                    and np.isclose(eta[g].max(), hi, atol=0.06)):
                raise SystemExit("the first block's two groups do not match the ranges "
                                 "decisions.md reports for them")
        fill.update(prior=span(groups[0]), random=span(groups[1]))

    # ---- the storyboard: one dict per frame -------------------------------------------
    # e: evaluations shown; dec: a decision on screen; flash / gflash: frames since a method
    # found the optimum.
    frames = []

    def emit(e, **kw):
        frames.append({"e": e, "dec": None, "intro": False, "flash": None, "gflash": None,
                       "end": False, **kw})

    DEC = 30
    for _ in range(18):
        emit(0, intro=True)
    for _ in range(DEC):
        emit(0, dec=0)
    for e in range(1, n + 1):
        emit(e)
        if e == found:
            for t in range(24):
                emit(e, flash=t)
        if e in cuts:
            for _ in range(DEC):
                emit(e, dec=cuts.index(e) + 1)
        if e == gfound:
            for t in range(24):
                emit(e, gflash=t)
    for _ in range(36):
        emit(n, end=True)

    # ---- the figure ---------------------------------------------------------------------
    fig, (axp, axg, cax, axr), cap, info = layout()
    Q = np.c_[np.clip(P[:, 0], EXT[0], EXT[1] - 1e-6), np.clip(P[:, 1], EXT[2], EXT[3] - 1e-6)]
    s = np.log10(eta.max() * 1.01 / eta)  # higher is better and > 0, so the max-landscape applies
    g = landscape(Q, s, EXT, "median", n=N_GRID)
    norm = LogNorm(eta.min(), np.median(eta))
    for ax in (axp, axg):
        im = ax.imshow(np.where(g > 0, eta.max() * 1.01 / 10 ** g, np.nan), origin="lower",
                       extent=EXT, aspect="auto", cmap=SCALE, norm=norm, interpolation="nearest",
                       zorder=0)
        ax.set(xlim=EXT[:2], ylim=EXT[2:], xlabel="Molecular weight, mole-fraction weighted")
    axp.set_ylabel("H-bond donors, mole-fraction weighted")
    axg.set_yticklabels([])
    stars = [ax.plot(*Q[best], marker="*", ms=15, mfc="white", mec="black", mew=0.6, zorder=8)[0]
             for ax in (axp, axg)]
    pulses = [ax.plot([], [], "o", mfc="none", mec=GOLD, mew=2.0, zorder=7)[0] for ax in (axp, axg)]
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("Lowest viscosity in the cell (cP)", fontsize=7)
    cb.set_ticks([t for t in (0.5, 0.75, 1, 1.5, 2) if norm.vmin <= t <= norm.vmax])
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    cb.ax.minorticks_off()

    phase = lambda i: sum(i >= c for c in cuts)  # noqa: E731  evaluation i (0-based) -> block
    walk = [Walker(axp, Q, idx, [AGENT[phase(i)] for i in range(n)]),
            Walker(axg, Q, gidx, [GREY] * n)]
    dots = [RANDOM if n_prior <= i < cuts[0] else AGENT[phase(i)] for i in range(n)]
    ra, rg = rank(-eta, idx), rank(-eta, gidx)
    climbed = climb(axr, ra, rg, N, dots, {c: f"decision {k + 2}" for k, c in enumerate(cuts)},
                    found, gfound, xlabel=f"Evaluations (first {END} of 250)")
    sd = Q.std(0)
    da, dg = (w.distance(Q[best], sd) for w in walk)
    wa, wg = (w.width(sd) for w in walk)
    late_a, late_g = np.median(ra[cuts[0]:]), np.median(rg[cuts[0]:])

    names = {"CCC(C)=O": "2-butanone", "CCC(C)(C)O": "tert-amyl alcohol"}
    who = " with ".join(f"{names.get(c, c)} ({f:.0%})" for c, f in
                        sorted(zip(smiles[best], x[best]), key=lambda t: -t[1]))
    ls_ = lambda v: ", ".join("never" if h is None else str(h) for h in sorted(  # noqa: E731
        v, key=lambda h: (h is None, h)))

    def draw(fi):
        f = frames[fi]
        e = f["e"]
        for w in walk:
            w.at(e)
        climbed(e)
        ab = eta[idx[:e]].min() if e else None
        gb = eta[gidx[:e]].min() if e else None
        fmt = lambda v, ok: "–" if v is None else f"{v:.3f} cP" + ("  (optimum)" if ok else "")  # noqa: E731,E501
        info[axp].set_text(f"Best so far: {fmt(ab, e >= found)}")
        info[axg].set_text(f"Best so far: {fmt(gb, gfound is not None and e >= gfound)}")
        for k, (star, done) in enumerate(((stars[0], e >= found),
                                          (stars[1], gfound is not None and e >= gfound))):
            star.set_markerfacecolor(GOLD if done else "white")
            star.set_markersize(19 if done else 15)
            fl = f["flash"] if k == 0 else f["gflash"]
            if fl is not None:
                t = (fl % 12) / 12
                pulses[k].set_data([Q[best, 0]], [Q[best, 1]])
                pulses[k].set_markersize(14 + 30 * t)
                pulses[k].set_alpha(1 - t)
            else:
                pulses[k].set_data([], [])

        di = f["dec"]
        if f["intro"]:
            txt = (f"Binary mixtures: minimize viscosity at room temperature ({N:,} candidates, "
                   f"{len({m for s_ in smiles for m in s_})} components).\nEach map: molecular "
                   "weight across, H-bond donors up (both mole-fraction weighted), each cell "
                   "coloured by its lowest-viscosity mixture.\nDot: the moving average of the "
                   "method's picks;  ellipse: their spread (1 s.d.);  line: the dot's path.\nNo "
                   "measurements yet. The agent reads what the variables mean; GP-BO sees only "
                   "numbers.")
        elif di is not None:
            d = DECISIONS[di]
            txt = (f"Decision {di + 1}, " + ("before any measurement" if d["after"] == 0 else
                                             f"at evaluation {d['after']}") + "\n"
                   + (f"Observes:  {d['observe'].format(**fill)}\n" if "observe" in d else "")
                   + f"Infers from chemistry:  {d['reason']}\nDesigns:  {d['design']}")
        elif f["flash"] is not None:
            txt = (f"Evaluation {found}: {who}, {eta[best]:.3f} cP: the pool optimum.\n"
                   f"GP-BO's best at this point: {gb:.2f} cP.")
        elif f["gflash"] is not None:
            txt = (f"Evaluation {gfound}: GP-BO, with no chemistry, reaches the same optimum, "
                   f"{gfound - found} evaluations after the agent.")
        elif f["end"]:
            txt = (f"After decision 2 (evaluations {cuts[0] + 1}–{n}), the agent's median pick is in "
                   f"the pool's top {late_a:.1f}%; GP-BO's in its top {late_g:.0f}%.\n"
                   f"This seed: the agent reaches the optimum at evaluation {found}; GP-BO "
                   + (f"at {gfound}." if gfound else f"not in the {n} shown (it takes {gever}).")
                   + f"\nAll seeds: the grey-box agent at {ls_(first_hits(a.results, METHOD))}; "
                   f"GP-BO at {ls_(first_hits(a.results, 'gp_bo'))}.")
        else:
            txt = cfg["after"][phase(e - 1)]
        cap.set_text(txt)
        return ()

    last = slice(n - 30, n)
    summary = dict(frames=len(frames), optimum=float(eta.min()), agent=found, gpbo=gfound,
                   gpbo_full=gever, groups=fill,
                   late=(round(float(late_a), 2), round(float(late_g), 1)),
                   # how far each average sits from the optimum, and how wide the picks are, over
                   # the last 30 evaluations shown, in pool s.d.
                   dist_last30=(round(float(da[last].mean()), 2), round(float(dg[last].mean()), 2)),
                   spread_last30=(round(float(wa[last].mean()), 2), round(float(wg[last].mean()), 2)),
                   at_end=(round(float(da[-1]), 2), round(float(dg[-1]), 2),
                           round(float(wa[-1]), 2), round(float(wg[-1]), 2)))
    return fig, draw, frames, summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--results", type=Path, default=Path("benchmark/graybox"))
    p.add_argument("--out", type=Path, default=Path("benchmark/figures/mix/mix.gif"))
    p.add_argument("--seed", type=int, default=1, choices=sorted(CAMPAIGNS))
    a = p.parse_args(argv)
    fig, draw, frames, summary = build(a)
    save(fig, draw, frames, a.out)
    print(a.out, summary)


if __name__ == "__main__":
    main()
