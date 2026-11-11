"""Analyse the axis study v3 sweep: which axes differentiate the baselines.

    python -m aobench.baselines.axis_v3_analyze RUNS/axis_v3

Writes scores.csv, effects.csv and report.md (plus axis_profiles.png) into the run folder.

Score (see tasks/AXIS_V3_PREDICTIONS.md): log10 regret of the noise-free value at
the best-observed point, floored at 1e-8, minus random search's median on that
task, at each checkpoint (10, 30, 100, 300 * d).

For each optimizer and axis, at each checkpoint:
- **rel**: the optimizer's score minus the mean of all optimizers on that task,
  averaged per axis level over everything else. This takes out how hard the task
  is for everyone.
- **p**: a permutation test of the spread of those level means. Level labels are
  shuffled among tasks of the same function (for structure, among functions),
  and Holm-corrected over the 4 axes per optimizer and checkpoint.
- **"degrades"**: high minus low rel ≥ 0.5 decades, the mean rank worsens by
  ≥ 1, and Holm p < 0.05. **"improves"** is the mirror image.

An axis **differentiates** if at least one pair of optimizers swaps order between
its low and high level by ≥ 0.5 decades, and both have a Holm-significant effect
on that axis.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..tasks.axis_v3 import CHECKPOINTS_PER_DIM, axis_v3_cells

AXES = ["structure", "conditioning", "dimension", "noise"]
LEVELS = {"structure": ["unimodal", "adequate", "weak"], "conditioning": ["low", "medium", "high"],
          "dimension": ["low", "medium", "high"], "noise": ["low", "medium", "high"]}
FLOOR = 1e-8
N_PERM = 2000


def load(root: Path) -> pd.DataFrame:
    cells = {c["key"]: c for c in axis_v3_cells()}
    rows = []
    for meta_path in root.glob("v3_*/*/seed_*/meta.json"):
        meta = json.loads(meta_path.read_text())
        key, d = meta["task_name"], meta["task_dim"]
        with open(meta_path.with_name("trace.csv")) as f:
            trace = [(int(r["eval_idx"]), float(r["y_best_so_far"])) for r in csv.DictReader(f)]
        for k in CHECKPOINTS_PER_DIM:
            best = [b for i, b in trace if i <= k * d]
            regret = max(best[-1] - meta["task_f_opt"], FLOOR) if best else np.nan
            c = cells[key]
            rows.append(dict(key=key, fname=c["fname"], opt=meta["optimizer_name"], seed=meta["seed"],
                             ckpt=k, lr=np.log10(regret), restarts=meta.get("n_restarts"),
                             **{a: c[a] for a in AXES}))
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def spread(values, labels, levels):
    return np.var([values[labels == lv].mean() for lv in levels])


def effects(df: pd.DataFrame, rng) -> pd.DataFrame:
    out = []
    for ck, g in df.groupby("ckpt"):
        t = g.groupby(["key", "opt"]).score.mean().unstack()
        rel = t.sub(t.mean(axis=1), axis=0)
        rank = g.groupby(["key", "opt"])["rank"].mean().unstack()
        info = g.drop_duplicates("key").set_index("key").loc[t.index]
        for opt in t.columns:
            ps, rec = [], []
            for axis in AXES:
                lv = LEVELS[axis]
                labels = info[axis].to_numpy()
                y = rel[opt].to_numpy()
                stat = spread(y, labels, lv)
                if axis == "structure":  # shuffle level labels among the 9 functions
                    fn = info["fname"].to_numpy()
                    fnames = np.unique(fn)
                    f_level = info.groupby("fname").structure.first()
                    perm_stats = []
                    for _ in range(N_PERM):
                        mapping = dict(zip(fnames, rng.permutation(f_level.loc[fnames].to_numpy())))
                        perm_stats.append(spread(y, np.array([mapping[f] for f in fn]), lv))
                else:  # shuffle level labels within each function
                    blocks = [np.flatnonzero(info["fname"].to_numpy() == f) for f in info["fname"].unique()]
                    perm_stats = []
                    for _ in range(N_PERM):
                        lab = labels.copy()
                        for b in blocks:
                            lab[b] = rng.permutation(lab[b])
                        perm_stats.append(spread(y, lab, lv))
                p = (np.sum(np.array(perm_stats) >= stat) + 1) / (N_PERM + 1)
                means = [y[labels == l].mean() for l in lv]
                rk = [rank[opt].to_numpy()[labels == l].mean() for l in lv]
                ps.append(p)
                rec.append(dict(ckpt=ck, opt=opt, axis=axis, rel_low=means[0], rel_mid=means[1],
                                rel_high=means[2], rank_low=rk[0], rank_mid=rk[1], rank_high=rk[2], p=p))
            for r, ph in zip(rec, holm(ps)):
                r["p_holm"] = ph
                delta, drank = r["rel_high"] - r["rel_low"], r["rank_high"] - r["rank_low"]
                r["verdict"] = ("degrades" if delta >= 0.5 and drank >= 1 and ph < 0.05 else
                                "improves" if delta <= -0.5 and drank <= -1 and ph < 0.05 else "robust")
                out.append(r)
    return pd.DataFrame(out)


def differentiating(eff: pd.DataFrame, ck: int) -> dict:
    res = {}
    e = eff[eff.ckpt == ck].set_index(["axis", "opt"])
    for axis in AXES:
        sig = [o for o in e.loc[axis].index if e.loc[(axis, o), "p_holm"] < 0.05]
        swaps = []
        for i, a in enumerate(sig):
            for b in sig[i + 1:]:
                lo = e.loc[(axis, a), "rel_low"] - e.loc[(axis, b), "rel_low"]
                hi = e.loc[(axis, a), "rel_high"] - e.loc[(axis, b), "rel_high"]
                if np.sign(lo) != np.sign(hi) and abs(lo - hi) >= 0.5:
                    swaps.append((a, b, round(lo, 2), round(hi, 2)))
        res[axis] = swaps
    return res


def main():
    root = Path(sys.argv[1])
    df = load(root)
    rs = df[df.opt == "random_search"].groupby(["key", "ckpt"]).lr.median()
    df["score"] = df.lr - pd.Series(list(zip(df.key, df.ckpt))).map(rs).to_numpy()
    df["rank"] = df.groupby(["key", "seed", "ckpt"]).lr.rank()
    df.to_csv(root / "scores.csv", index=False)
    rng = np.random.default_rng(0)
    eff = effects(df, rng)
    eff.to_csv(root / "effects.csv", index=False)

    final = max(CHECKPOINTS_PER_DIM)
    lines = [f"# Axis study v3: results\n", f"{df.groupby(['key','opt','seed']).ngroups} campaigns, "
             f"{df.key.nunique()} tasks, optimizers: {', '.join(sorted(df.opt.unique()))}.\n"]
    # Budget checks (plan, step 3)
    g = df[df.ckpt == final]
    easy = g[(g.structure == "unimodal") & (g.conditioning == "low") & (g.noise == "low")]
    best_easy = easy.groupby(["key", "opt"]).lr.median().groupby("key").min()
    lead = g.groupby(["key", "opt"]).score.median().groupby("key").min()
    lines += ["## Budget checks at 300·d\n",
              f"- (a) Easy cells (unimodal, conditioning 1, noiseless) where the best optimizer's median "
              f"regret ≤ 1e-8: {int((best_easy <= -8 + 1e-9).sum())} of {len(best_easy)}.",
              f"- (b) Tasks where the best optimizer's median is ≥ 1 decade ahead of random search: "
              f"{int((lead <= -1).sum())} of {len(lead)} ({(lead <= -1).mean():.0%}).\n"]
    for ck in sorted(CHECKPOINTS_PER_DIM, reverse=True):
        e = eff[eff.ckpt == ck]
        lines += [f"## Effects at {ck}·d\n",
                  "rel = score minus the all-optimizer mean on that task (decades; negative is better). "
                  "Levels low / mid / high (structure: unimodal / adequate / weak).\n",
                  "| axis | optimizer | rel low | rel mid | rel high | rank low → high | Holm p | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in e.sort_values(["axis", "opt"]).iterrows():
            lines.append(f"| {r.axis} | {r.opt} | {r.rel_low:+.2f} | {r.rel_mid:+.2f} | {r.rel_high:+.2f} | "
                         f"{r.rank_low:.1f} → {r.rank_high:.1f} | {r.p_holm:.3g} | {r.verdict} |")
        diff = differentiating(eff, ck)
        lines += ["", "Differentiating axes (significant optimizers whose order swaps between low and high):\n"]
        for axis, swaps in diff.items():
            lines.append(f"- **{axis}**: " + ("; ".join(f"{a} vs {b} ({lo:+} → {hi:+})" for a, b, lo, hi in swaps)
                                              if swaps else "none"))
        lines.append("")
    (root / "report.md").write_text("\n".join(lines))
    try:
        plot(eff[eff.ckpt == final], root / "axis_profiles.png")
    except Exception as exc:  # plotting is optional
        print("plot skipped:", exc)
    print((root / "report.md").read_text())


def plot(e, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    opts = sorted(e.opt.unique())
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, axis in zip(axs, AXES):
        for o in opts:
            r = e[(e.axis == axis) & (e.opt == o)].iloc[0]
            ax.plot([0, 1, 2], [r.rel_low, r.rel_mid, r.rel_high], marker="o", label=o)
        ax.set_xticks([0, 1, 2], LEVELS[axis])
        ax.set_title(axis)
        ax.axhline(0, color="grey", lw=0.5)
    axs[0].set_ylabel("log10 regret vs all-optimizer mean (lower is better)")
    axs[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
