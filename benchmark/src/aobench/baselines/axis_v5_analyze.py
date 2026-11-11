"""Analyse the axis study v5 sweep: does every solver fail somewhere, and do the predicted reversals hold?

    python -m aobench.baselines.axis_v5_analyze RUNS/axis_v5

Writes scores.csv, pairs.csv and report.md into the run folder. Definitions
(frozen in tasks/AXIS_V5_PREDICTIONS.md, unchanged from v4):

- score: log10 regret at 300*d (noise-free f at the best-observed point), floored at 1e-8;
- advantage: median over seeds of score(sobol) - score(solver); a solver fails on a cell
  if its advantage is < 0.1 decades, and collapses on a group if it fails on >= 80% of its cells;
- A beats B in a cell: paired Wilcoxon signed-rank over seeds, Holm over all solver pairs
  in the cell, p < 0.05 and median(score_B - score_A) > 0;
- A beats B on a group: in >= 2/3 of its cells, and B never beats A there;
- rank reversal: A beats B on one group, B beats A on another.

Some predictions apply to part of a group only (weierstrass at d >= 20); they are
written as "group@dims". The dimensions analysed are those present in the run.
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

from ..tasks.axis_v5 import CHECKPOINTS_PER_DIM, GROUPS as TASK_GROUPS, axis_v5_cells
from .axis_v5_sweep import OPTIMIZERS, SEEDS

FLOOR = 1e-8
FAIL = 0.1
GROUPS = list(TASK_GROUPS)
SOLVERS = [o for o in OPTIMIZERS if o != "sobol"]

# Frozen 2026-11-10 (tasks/AXIS_V5_PREDICTIONS.md). A scope is "group" or "group@d1,d2".
HYPOTHESES = {  # id: (solver that collapses, scope, [(winner, loser, scope), ...])
    "H1": ("nelder_mead", "plateau", []),
    "H2": ("lbfgsb", "noise_moderate", []),
    "H3": ("cobyla", "noise_severe", []),
    "H4": ("cma_es", "katsuura", [("nelder_mead", "cma_es", "katsuura"), ("cobyla", "cma_es", "katsuura")]),
    "H5": ("cma_es", "weierstrass@20,40", [("cobyla", "cma_es", "weierstrass@20,40")]),
    "H6": ("turbo", "illcond", [("lbfgsb", "turbo", "illcond")]),
}
PREDICTED_REVERSALS = {  # id: (A, B, scopes where A beats B, scopes where B beats A)
    "R1": ("lbfgsb", "turbo", ["illcond"], ["noise_moderate", "noise_severe"]),
    "R2": ("cma_es", "turbo", ["illcond"], ["katsuura", "weierstrass@20,40"]),
    "R3": ("nelder_mead", "cma_es", ["katsuura"], ["plateau"]),
    "R4": ("cobyla", "cma_es", ["katsuura", "weierstrass@20,40"], ["noise_severe", "plateau"]),
    "R5": ("lbfgsb", "cma_es", ["illcond"], ["noise_moderate", "noise_severe"]),
    "R6": ("lbfgsb", "cobyla", ["illcond"], ["noise_moderate"]),
    "R7": ("lbfgsb", "sobol", ["illcond"], ["noise_moderate"]),
    "R8": ("nelder_mead", "sobol", ["illcond"], ["plateau"]),
}
SECONDARY_COLLAPSE = {  # v4 replications and TuRBO extras: (collapses on, does not collapse on)
    "lbfgsb": (["noise_severe", "plateau", "katsuura", "weierstrass"], ["illcond"]),
    "nelder_mead": (["noise_moderate", "noise_severe"], ["illcond"]),
    "cobyla": ([], ["illcond", "noise_moderate", "plateau"]),
    "cma_es": ([], ["illcond", "noise_moderate", "noise_severe", "plateau"]),
    "turbo": ([], ["noise_moderate"]),
}
NAME = {"sobol": "Sobol", "lbfgsb": "L-BFGS-B", "nelder_mead": "Nelder–Mead", "cobyla": "COBYLA",
        "cma_es": "CMA-ES", "turbo": "TuRBO"}


def md(df: pd.DataFrame) -> str:
    """A DataFrame as a markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    rows = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, r in df.iterrows():
        rows.append(f"| {idx} | " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(rows)


def load(root: Path) -> pd.DataFrame:
    cells = {c["key"]: c for c in axis_v5_cells()}
    rows = []
    for meta_path in root.glob("v5_*/*/seed_*/meta.json"):
        meta = json.loads(meta_path.read_text())
        key, d = meta["task_name"], meta["task_dim"]
        with open(meta_path.with_name("trace.csv")) as f:
            trace = [(int(r["eval_idx"]), float(r["y_best_so_far"])) for r in csv.DictReader(f)]
        c = cells[key]
        for k in CHECKPOINTS_PER_DIM:
            best = [b for i, b in trace if i <= k * d]
            regret = max(best[-1] - meta["task_f_opt"], FLOOR) if best else np.nan
            rows.append(dict(key=key, group=c["group"], fname=c["fname"], dim=d, beta=c["beta"],
                             opt=meta["optimizer_name"], seed=meta["seed"], ckpt=k,
                             score=np.log10(regret), restarts=meta.get("n_restarts"),
                             evals=meta["evaluations_used"], wall=meta["total_wall_seconds"]))
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
            recs.append(dict(key=key, group=g.group.iloc[0], dim=int(g.dim.iloc[0]), a=a, b=b, n=len(diff),
                             median_b_minus_a=float(np.median(diff)) if len(diff) else np.nan, p=p))
        for r, ph in zip(recs, holm([r["p"] for r in recs])):
            r["p_holm"] = ph
            r["winner"] = (r["a"] if ph < 0.05 and r["median_b_minus_a"] > 0 else
                           r["b"] if ph < 0.05 and r["median_b_minus_a"] < 0 else "")
            out.append(r)
    return pd.DataFrame(out)


def parse_scope(scope: str):
    group, _, dims = scope.partition("@")
    return group, ({int(d) for d in dims.split(",")} if dims else None)


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    group, dims = parse_scope(scope)
    m = df.group == group
    return m & df.dim.isin(dims) if dims else m


def beats(pairs: pd.DataFrame, a: str, b: str, scope: str) -> tuple[bool, int, int, int]:
    lo, hi = sorted([a, b])
    g = pairs[scope_mask(pairs, scope) & (pairs.a == lo) & (pairs.b == hi)]
    wins, losses, n = int((g.winner == a).sum()), int((g.winner == b).sum()), len(g)
    return (n > 0 and wins >= 2 * n / 3 and losses == 0), wins, losses, n


def fmt_beats(pairs, a, b, scope) -> tuple[str, bool]:
    ok, w, l, n = beats(pairs, a, b, scope)
    return f"{scope} {w}–{l}/{n} {'✓' if ok else '✗'}", ok


def collapse(adv: pd.DataFrame, opt: str, scope: str) -> tuple[bool, int, int]:
    s = adv[scope_mask(adv, scope)][opt].dropna()
    n_fail = int((s < FAIL).sum())
    return (len(s) > 0 and n_fail >= 0.8 * len(s)), n_fail, len(s)


def main():
    root = Path(sys.argv[1])
    df = load(root)
    df.to_csv(root / "scores.csv", index=False)
    final = df[df.ckpt == CHECKPOINTS_PER_DIM[-1]].copy()
    dims = sorted(final.dim.unique())
    cells = axis_v5_cells(dims)
    expected = len(cells) * len(OPTIMIZERS) * len(SEEDS)
    pairs = pair_tests(final)
    pairs.to_csv(root / "pairs.csv", index=False)
    final["rank"] = final.groupby(["key", "seed"]).score.rank(method="average")
    wide = final.pivot_table(index=["group", "dim", "key", "seed"], columns="opt", values="score")
    adv = (-wide.sub(wide["sobol"], axis=0)).groupby(["group", "dim", "key"]).median().drop(columns="sobol")
    adv = adv.reset_index().set_index("key")
    cell_rank = final.pivot_table(index="key", columns="opt", values="rank", aggfunc="mean")

    L = ["# Axis study v5 results\n",
         f"{len(final)} campaigns loaded (expected {expected}: {len(cells)} cells × {len(OPTIMIZERS)} solvers × "
         f"{len(SEEDS)} seeds), dimensions {', '.join(map(str, dims))}, score = log₁₀ regret at 300·d. "
         f"Seeds {SEEDS.start}–{SEEDS.stop - 1} = BBOB instances {SEEDS.start + 1}–{SEEDS.stop}.\n"]
    have = final.groupby(["key", "opt"]).size()
    missing = [f"{k} / {o}: {len(SEEDS) - have.get((k, o), 0)} missing"
               for k in (c["key"] for c in cells) for o in OPTIMIZERS if have.get((k, o), 0) < len(SEEDS)]
    if missing:
        L += ["**Incomplete cells** (tests use the seeds present for both solvers):\n"] + \
             [f"- {m}" for m in missing] + [""]

    # 1. Failures per solver.
    L += ["## 1. Where each solver fails\n",
          f"Fail = advantage over Sobol (median over seeds of score(Sobol) − score(solver)) < {FAIL} decades. "
          "Mean rank is within the cell (1 = best of 6). Best = lowest mean rank in the cell; "
          "'beaten by' = solvers that beat it there (Holm p < 0.05).\n"]
    for opt in SOLVERS:
        fails = adv[adv[opt] < FAIL].sort_values(["group", "dim"])
        L.append(f"### {NAME[opt]}: fails on {len(fails)}/{adv[opt].notna().sum()} cells\n")
        if fails.empty:
            L += ["(no failed cell)\n"]
            continue
        L += ["| Cell | Advantage | Mean rank | Best in cell | Beaten by |", "|---|---|---|---|---|"]
        for key, r in fails.iterrows():
            best = cell_rank.loc[key].idxmin()
            p = pairs[(pairs.key == key) & ((pairs.a == opt) | (pairs.b == opt))]
            by = sorted({w for w in p.winner if w and w != opt})
            L.append(f"| {key} | {r[opt]:+.2f} | {cell_rank.loc[key, opt]:.2f} | {NAME[best]} "
                     f"({cell_rank.loc[key, best]:.2f}) | {', '.join(NAME[b] for b in by) or '—'} |")
        L.append("")

    share = pd.DataFrame({o: {g: collapse(adv, o, g)[1] / max(collapse(adv, o, g)[2], 1) for g in GROUPS}
                          for o in SOLVERS})
    L += ["### Share of cells failed, per group\n", md(share.round(2)), "",
          "### Median advantage over Sobol per group (decades)\n",
          md(adv.groupby("group")[SOLVERS].median().reindex(GROUPS).round(2)), ""]
    mr = final.pivot_table(index="group", columns="opt", values="rank", aggfunc="mean").reindex(GROUPS)[OPTIMIZERS]
    L += ["### Mean rank per group (1 = best)\n", md(mr.round(2)), ""]

    # 2. Verdicts.
    L += ["## 2. Verdicts on the frozen predictions\n", "### Hypotheses (who fails, who wins there)\n",
          "| | Collapse predicted | Observed | Wins predicted | Verdict |", "|---|---|---|---|---|"]
    verdict = {}
    for hid, (opt, scope, wins) in HYPOTHESES.items():
        ok, nf, n = collapse(adv, opt, scope)
        parts, all_w = [], True
        for a, b, sc in wins:
            s, w_ok = fmt_beats(pairs, a, b, sc)
            parts.append(f"{NAME[a]} > {NAME[b]}: {s}")
            all_w &= w_ok
        verdict[hid] = (ok, all_w)
        v = ("**confirmed**" if ok and all_w else "collapse confirmed, wins not" if ok else
             "collapse not confirmed" + (", wins confirmed" if wins and all_w else ""))
        L.append(f"| {hid} | {NAME[opt]} on {scope} | fails {nf}/{n} cells {'✓' if ok else '✗'} | "
                 f"{'; '.join(parts) or '—'} | {v} |")
    L += ["", "### Predicted rank reversals\n", "| | Pair | A beats B on | B beats A on | Verdict |",
          "|---|---|---|---|---|"]
    n_conf = 0
    for rid, (a, b, ab, ba) in PREDICTED_REVERSALS.items():
        r1 = [fmt_beats(pairs, a, b, s) for s in ab]
        r2 = [fmt_beats(pairs, b, a, s) for s in ba]
        conf = any(o for _, o in r1) and any(o for _, o in r2)
        n_conf += conf
        L.append(f"| {rid} | {NAME[a]} vs {NAME[b]} | {'; '.join(s for s, _ in r1)} | "
                 f"{'; '.join(s for s, _ in r2)} | {'**confirmed**' if conf else 'not confirmed'} |")
    L.append("\n(Confirmed = A beats B on at least one listed scope and B beats A on at least one.)\n")
    L += ["### Secondary predictions (v4 replications, TuRBO extras)\n",
          "| Solver | Predicted collapse: observed | Predicted no collapse: observed |", "|---|---|---|"]
    for opt, (yes, no) in SECONDARY_COLLAPSE.items():
        def f(gs, want):
            return "; ".join(f"{g} {collapse(adv, opt, g)[1]}/{collapse(adv, opt, g)[2]} "
                             f"{'✓' if collapse(adv, opt, g)[0] == want else '✗'}" for g in gs) or "—"
        L.append(f"| {NAME[opt]} | {f(yes, True)} | {f(no, False)} |")
    L.append("")

    found = []
    for a, b in itertools.combinations(OPTIMIZERS, 2):
        ab = [g for g in GROUPS if beats(pairs, a, b, g)[0]]
        ba = [g for g in GROUPS if beats(pairs, b, a, g)[0]]
        if ab and ba:
            found.append(f"- **{NAME[a]} vs {NAME[b]}**: {NAME[a]} wins on {', '.join(ab)}; "
                         f"{NAME[b]} wins on {', '.join(ba)}")
    L += ["### All rank reversals found (whole groups)\n"] + (found or ["(none)"]) + [""]

    L += ["### Who beats whom, per group\n",
          "Cell = wins–losses of row over column across the group's cells (Holm p < 0.05). "
          "**Bold** = beats on the group (≥ ⅔ of cells, no losses).\n"]
    for grp in GROUPS:
        tab = pd.DataFrame(index=[NAME[o] for o in OPTIMIZERS], columns=[NAME[o] for o in OPTIMIZERS], dtype=object)
        for a in OPTIMIZERS:
            for b in OPTIMIZERS:
                ok, w, l, n = beats(pairs, a, b, grp) if a != b else (False, 0, 0, 0)
                tab.loc[NAME[a], NAME[b]] = "—" if a == b else (f"**{w}–{l}**" if ok else f"{w}–{l}")
        L += [f"#### {grp} ({pairs[pairs.group == grp].key.nunique()} cells)\n", md(tab), ""]

    # 3. Honest note.
    fails_any = {o: bool((adv[o] < FAIL).any()) for o in SOLVERS}
    collapses_any = {o: [g for g in GROUPS if collapse(adv, o, g)[0]] for o in SOLVERS}
    failed_h = [h for h, (c, w) in verdict.items() if not (c and w)]
    L += ["## 3. Notes on failed predictions\n",
          f"- Hypotheses confirmed in full: {len(verdict) - len(failed_h)}/{len(verdict)}"
          + (f"; not: {', '.join(failed_h)}." if failed_h else "."),
          f"- Predicted reversals confirmed: {n_conf}/{len(PREDICTED_REVERSALS)}.",
          "- Every non-reference solver fails on at least one cell: "
          + ("yes" if all(fails_any.values()) else "no (" + ", ".join(NAME[o] for o, v in fails_any.items() if not v) + ")"),
          "- Every non-reference solver collapses on at least one group: "
          + ("yes" if all(collapses_any.values()) else
             "no (" + ", ".join(NAME[o] for o, v in collapses_any.items() if not v) + ")")]
    for hid in failed_h:
        opt, scope, wins = HYPOTHESES[hid]
        group, sdims = parse_scope(scope)
        lost = sorted(sdims - set(dims)) if sdims else []
        sub = adv[scope_mask(adv, scope)]
        fn = {c["key"]: c["fname"] for c in cells}
        per = "; ".join(f"{fn[k]} d = {r.dim}: {r[opt]:+.2f}, mean rank {cell_rank.loc[k, opt]:.2f}, "
                        f"best {NAME[cell_rank.loc[k].idxmin()]}" for k, r in sub.iterrows())
        c, w = verdict[hid]
        L.append(f"- **{hid} ({NAME[opt]} on {scope})**: collapse {'held' if c else 'did not hold'}"
                 + (f", predicted winners {'held' if w else 'did not hold'}" if wins else "")
                 + f". Advantage per cell: {per}."
                 + (f" Cells at d = {', '.join(map(str, lost))} were not run, so the scope is smaller than "
                    "frozen." if lost else ""))
    predicted = {(o, parse_scope(sc)[0]) for o, sc, _ in HYPOTHESES.values()} | \
                {(o, g) for o, (yes, _) in SECONDARY_COLLAPSE.items() for g in yes}
    post_hoc = [(o, g) for o in SOLVERS for g in collapses_any[o] if (o, g) not in predicted]
    L.append("- Collapses observed but not predicted (post hoc; they need fresh instances to count): "
             + ("; ".join(f"{NAME[o]} on {g} ({collapse(adv, o, g)[1]}/{collapse(adv, o, g)[2]})"
                          for o, g in post_hoc) or "none") + ".")
    for opt, hids in [("cma_es", ["H4", "H5"]), ("turbo", ["H6"])]:
        held = [h for h in hids if verdict[h][0]]
        ph = [g for o, g in post_hoc if o == opt]
        L.append(f"- **Does {NAME[opt]} now have a confirmed failure?** "
                 + (f"Yes: {', '.join(held)} held on instances not used to form it "
                    f"({', '.join(HYPOTHESES[h][1] for h in held)})." if held else
                    "No: its frozen collapse prediction did not hold.")
                 + (f" Unpredicted collapse on {', '.join(ph)}: a candidate for the next study, not a "
                    "confirmation." if ph else ""))
    L.append("")
    conf = {rid: any(beats(pairs, a, b, s)[0] for s in ab) and any(beats(pairs, b, a, s)[0] for s in ba)
            for rid, (a, b, ab, ba) in PREDICTED_REVERSALS.items()}
    L += ["## Success criterion (frozen)\n",
          "- H1–H6 collapses all hold: " + ("met" if all(c for c, _ in verdict.values()) else
                                           "not met (" + ", ".join(h for h, (c, _) in verdict.items() if not c) + ")"),
          f"- R1 and R2 (TuRBO reversals) both confirmed: {'met' if conf['R1'] and conf['R2'] else 'not met'}",
          f"- At least 5 of R1–R8 confirmed: {n_conf}/8, {'met' if n_conf >= 5 else 'not met'}", ""]
    wall = final.groupby(["opt", "dim"]).wall.agg(["count", "median", "max"])
    wall[["median", "max"]] /= 60
    L += ["## Wall time per campaign (minutes)\n", md(wall.round(1)), ""]
    (root / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
