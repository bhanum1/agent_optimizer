"""Analyse the axis study v4 sweep: are the predicted rank reversals real?

    python -m aobench.baselines.axis_v4_analyze RUNS/axis_v4

Writes scores.csv, pairs.csv and report.md into the run folder. Definitions
(frozen in tasks/AXIS_V4_PREDICTIONS.md):

- score: log10 regret at 300*d (noise-free f at the best-observed point), floored at 1e-8;
- A beats B in a cell: paired Wilcoxon signed-rank over seeds, Holm over the 10 pairs
  in the cell, p < 0.05 and median(score_B - score_A) > 0;
- A beats B on a group: in >= 2/3 of its cells, and B never beats A there;
- rank reversal: A beats B on one group, B beats A on another;
- advantage: median over seeds of score(sobol) - score(solver); collapse on a group:
  advantage < 0.1 in >= 80% of its cells.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..tasks.axis_v4 import CHECKPOINTS_PER_DIM, axis_v4_cells
from .axis_v4_sweep import OPTIMIZERS

FLOOR = 1e-8
GROUPS = ["illcond", "noise_moderate", "noise_severe", "highdim", "rugged", "plateau"]
PREDICTED_REVERSALS = {  # (A, B): (groups where A beats B, groups where B beats A)
    "R1": ("lbfgsb", "cma_es", ["illcond"], ["noise_moderate", "noise_severe", "rugged"]),
    "R2": ("lbfgsb", "cobyla", ["illcond"], ["noise_moderate"]),
    "R3": ("lbfgsb", "sobol", ["illcond"], ["noise_moderate"]),
    "R4": ("nelder_mead", "sobol", ["illcond"], ["noise_moderate"]),
}
PREDICTED_COLLAPSE = {
    "lbfgsb": (["noise_moderate", "noise_severe", "plateau"], ["illcond", "highdim"]),
    "nelder_mead": (["noise_moderate", "noise_severe", "rugged"], ["illcond"]),
    "cobyla": (["noise_severe"], ["illcond", "noise_moderate"]),
    "cma_es": (["plateau"], ["illcond", "noise_moderate", "noise_severe", "highdim", "rugged"]),
}


def md(df: pd.DataFrame) -> str:
    """A DataFrame as a markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    rows = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, r in df.iterrows():
        rows.append(f"| {idx} | " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(rows)


def analysis_group(cell) -> str:
    if cell["group"] == "noise":
        return "noise_moderate" if cell["beta"] < 0.1 else "noise_severe"
    return cell["group"]


def load(root: Path) -> pd.DataFrame:
    cells = {c["key"]: c for c in axis_v4_cells()}
    rows = []
    for meta_path in root.glob("v4_*/*/seed_*/meta.json"):
        meta = json.loads(meta_path.read_text())
        key, d = meta["task_name"], meta["task_dim"]
        with open(meta_path.with_name("trace.csv")) as f:
            trace = [(int(r["eval_idx"]), float(r["y_best_so_far"])) for r in csv.DictReader(f)]
        c = cells[key]
        for k in CHECKPOINTS_PER_DIM:
            best = [b for i, b in trace if i <= k * d]
            regret = max(best[-1] - meta["task_f_opt"], FLOOR) if best else np.nan
            rows.append(dict(key=key, group=analysis_group(c), fname=c["fname"], dim=d, beta=c["beta"],
                             opt=meta["optimizer_name"], seed=meta["seed"], ckpt=k,
                             score=np.log10(regret), restarts=meta.get("n_restarts")))
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def pair_tests(final: pd.DataFrame) -> pd.DataFrame:
    """Per cell and ordered pair (a < b alphabetically): who wins, Holm-corrected."""
    out = []
    for key, g in final.groupby("key"):
        wide = g.pivot(index="seed", columns="opt", values="score")
        recs = []
        for a, b in itertools.combinations(sorted(wide.columns), 2):
            diff = (wide[b] - wide[a]).dropna()  # > 0 means a is better
            if len(diff) == 0 or np.all(diff == 0):
                p = 1.0
            else:
                p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue
            recs.append(dict(key=key, group=g.group.iloc[0], a=a, b=b, n=len(diff),
                             median_b_minus_a=float(np.median(diff)) if len(diff) else np.nan, p=p))
        for r, ph in zip(recs, holm([r["p"] for r in recs])):
            r["p_holm"] = ph
            r["winner"] = (r["a"] if ph < 0.05 and r["median_b_minus_a"] > 0 else
                           r["b"] if ph < 0.05 and r["median_b_minus_a"] < 0 else "")
            out.append(r)
    return pd.DataFrame(out)


def beats_on_group(pairs: pd.DataFrame, a: str, b: str, group: str) -> tuple[bool, int, int, int]:
    lo, hi = sorted([a, b])
    g = pairs[(pairs.group == group) & (pairs.a == lo) & (pairs.b == hi)]
    wins, losses, n = int((g.winner == a).sum()), int((g.winner == b).sum()), len(g)
    return (n > 0 and wins >= 2 * n / 3 and losses == 0), wins, losses, n


def advantage(final: pd.DataFrame) -> pd.DataFrame:
    wide = final.pivot_table(index=["group", "key", "seed"], columns="opt", values="score")
    adv = (-wide.sub(wide["sobol"], axis=0)).groupby(["group", "key"]).median()
    return adv.drop(columns="sobol")


def main():
    root = Path(sys.argv[1])
    df = load(root)
    df.to_csv(root / "scores.csv", index=False)
    final = df[df.ckpt == CHECKPOINTS_PER_DIM[-1]]
    expected = len(axis_v4_cells()) * len(OPTIMIZERS) * final.seed.nunique()
    pairs = pair_tests(final)
    pairs.to_csv(root / "pairs.csv", index=False)
    adv = advantage(final)
    L = [f"# Axis study v4 results\n",
         f"{len(final)} campaigns loaded (expected {expected}), {final.seed.nunique()} seeds, score at 300·d.\n"]

    # Mean rank per group (rank within each cell x seed, 1 = best, ties averaged).
    final = final.assign(rank=final.groupby(["key", "seed"]).score.rank(method="average"))
    mr = final.pivot_table(index="group", columns="opt", values="rank", aggfunc="mean").reindex(GROUPS)
    L += ["## Mean rank per group (1 = best)\n", md(mr.round(2)), ""]
    med = final.pivot_table(index="group", columns="opt", values="score", aggfunc="median").reindex(GROUPS)
    L += ["## Median log₁₀ regret per group\n", md(med.round(2)), ""]

    # Pairwise group verdicts for all pairs.
    L += ["## Who beats whom, per group\n",
          "Cell = wins–losses of row over column across the group's cells (Holm p < 0.05). "
          "**Bold** = beats on the group (≥ ⅔ of cells, no losses).\n"]
    for grp in GROUPS:
        tab = pd.DataFrame(index=OPTIMIZERS, columns=OPTIMIZERS, dtype=object)
        for a in OPTIMIZERS:
            for b in OPTIMIZERS:
                if a == b:
                    tab.loc[a, b] = "—"
                    continue
                ok, w, l, n = beats_on_group(pairs, a, b, grp)
                tab.loc[a, b] = f"**{w}–{l}**" if ok else f"{w}–{l}"
        L += [f"### {grp} ({pairs[pairs.group == grp].key.nunique()} cells)\n", md(tab), ""]

    # All rank reversals found (data-driven, not just predicted ones).
    found = []
    for a, b in itertools.combinations(OPTIMIZERS, 2):
        ab = [g for g in GROUPS if beats_on_group(pairs, a, b, g)[0]]
        ba = [g for g in GROUPS if beats_on_group(pairs, b, a, g)[0]]
        if ab and ba:
            found.append(f"- **{a} vs {b}**: {a} wins on {', '.join(ab)}; {b} wins on {', '.join(ba)}")
    L += ["## Rank reversals found\n"] + (found or ["(none)"]) + [""]

    # Predicted reversals.
    L += ["## Predicted reversals\n", "| | Pair | A beats B on | B beats A on | Verdict |", "|---|---|---|---|---|"]
    n_conf = 0
    for rid, (a, b, ab, ba) in PREDICTED_REVERSALS.items():
        def fmt(x, y, groups):
            parts, any_ok = [], False
            for g in groups:
                ok, w, l, n = beats_on_group(pairs, x, y, g)
                any_ok |= ok
                parts.append(f"{g} {w}–{l}/{n} {'✓' if ok else '✗'}")
            return "; ".join(parts), any_ok
        s1, ok1 = fmt(a, b, ab)
        s2, ok2 = fmt(b, a, ba)
        conf = ok1 and ok2
        n_conf += conf
        L.append(f"| {rid} | {a} vs {b} | {s1} | {s2} | {'**confirmed**' if conf else 'not confirmed'} |")
    L.append("\n(A reversal is confirmed if A beats B on at least one listed group and B beats A on at least one.)\n")

    # Collapse.
    share = (adv < 0.1).groupby("group").mean().reindex(GROUPS)
    L += ["## Share of cells collapsed (advantage over Sobol < 0.1 decades)\n", md(share.round(2)), "",
          "## Median advantage over Sobol per group (decades)\n",
          md(adv.groupby("group").median().reindex(GROUPS).round(2)), "",
          "## Predicted collapses\n", "| Solver | Predicted collapse | Observed (≥ 80%) | Predicted no collapse, observed |",
          "|---|---|---|---|"]
    collapses_any = {}
    for opt, (yes, no) in PREDICTED_COLLAPSE.items():
        obs = [g for g in GROUPS if share.loc[g, opt] >= 0.8]
        collapses_any[opt] = bool(obs)
        L.append(f"| {opt} | {', '.join(yes) or 'none'} | {', '.join(obs) or 'none'} | "
                 f"{', '.join(f'{g} {share.loc[g, opt]:.0%}' for g in no)} |")
    ok_collapse = all(collapses_any[o] for o in ["lbfgsb", "nelder_mead", "cobyla"])
    L += ["", "## Success criterion\n",
          f"- Predicted reversals confirmed: {n_conf}/4 (need ≥ 3): {'met' if n_conf >= 3 else 'not met'}",
          f"- L-BFGS-B, Nelder–Mead and COBYLA each collapse on some group: {'met' if ok_collapse else 'not met'}",
          f"- CMA-ES collapses on plateau (amendment 1): {'yes' if share.loc['plateau', 'cma_es'] >= 0.8 else 'no'}", ""]
    (root / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
