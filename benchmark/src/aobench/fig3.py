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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .counterfactual import workspace  # noqa: E402
from .figures import METHODS, STYLE, WIDTH  # noqa: E402
from .tasks import make_task  # noqa: E402

FLOOR = 1e-8
AGENT = METHODS["agent"][1]
BASELINES = ["lbfgsb", "cobyla", "cma_es", "gpbo", "bandit", "sobol"]
BOLD = dict(fontfamily="STIXGeneral", fontweight="bold")  # STIX Two Text has no bold weight here
INK, MUTED, FAINT, RULE = "#1A1A1A", "#5F5F5F", "#9A9A9A", "#D9D9D9"
GREY = "#9A9A9A"  # the baseline
COLOURS = ["#D55E00", "#0072B2", "#009E73"]  # Okabe-Ito vermillion, blue, green: one per decision

# The task as the agent saw it, what was hidden from it, and each key decision: when, the evidence,
# the reasoning and the decision (from its rationale in decisions.md), its counterfactual branch,
# and that branch's label in panel b (what runs instead). `label_at` places a label by hand.
CAMPAIGNS = {
    "few_evals_10d/1": dict(
        task=r"Black box on $[-5, 5]^{10}$, 300 evaluations, 1% noise",
        hidden=r"Hidden from the agent: a sphere (BBOB $f_1$)",
        decisions=[
            dict(after=47, branch="gpbo_47", instead="restart GP-BO",
                 evidence="A separable quadratic fits all 47 points.",
                 reasoning="So its minimum can be computed, not searched.",
                 decision="Jump to the fitted minimum, then refine locally."),
            dict(after=126, branch="lq_126", instead="keep the trust region", label_at=(243, -4.72),
                 evidence="The trust region stalled; repeats agree to 5e-7.",
                 reasoning="Noise is negligible, so slopes can be measured.",
                 decision="Write its own finite-difference Newton method."),
        ],
        agent_label=(104, -5.1), base_label=(172, 0.95)),
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
        return "no effect on the final regret"
    k = round(abs(gain))
    return rf"$10^{{{k}}}\times$ {'lower' if gain > 0 else 'higher'} final regret"


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
    ax.text(x, y, str(number), ha="center", va="center", fontsize=size * 0.6, color=colour, zorder=7,
            **kw, **BOLD)


def panel_a(fig, ax, name, decisions):
    """Typeset in points from the top of the panel, so spacing stays even at any figure size."""
    ax.set_axis_off()
    cfg = CAMPAIGNS[name]
    h_pt = ax.get_position().height * fig.get_figheight() * 72
    y_of = lambda pt: 1 - pt / h_pt  # noqa: E731  distance from the top, in points, to axes units
    tr = ax.transAxes
    x_label, x_text = 0.115, 0.305

    ax.text(0.0, y_of(2), "TASK", ha="left", va="top", fontsize=6.0, color=FAINT, transform=tr)
    ax.text(x_label, y_of(1), cfg["task"], ha="left", va="top", fontsize=7.4, color=INK, transform=tr)
    ax.text(x_label, y_of(12), cfg["hidden"], ha="left", va="top", fontsize=6.8, color=MUTED,
            style="italic", transform=tr)
    ax.plot([0, 1], [y_of(27)] * 2, color=RULE, lw=0.6, transform=tr, clip_on=False)

    rows = (("EVIDENCE", "evidence", MUTED, {}), ("REASONING", "reasoning", MUTED, {}),
            ("DECISION", "decision", INK, BOLD))
    top, line, block = 37, 12.5, 66
    for i, d in enumerate(decisions):
        t = top + i * block
        badge(ax, 0.042, y_of(t + 7), i + 1, d["colour"], transform=tr)
        ax.text(0.042, y_of(t + 19), f"eval {d['after']}", ha="center", va="top", fontsize=6.0, color=FAINT,
                transform=tr)
        for j, (label, key, colour, font) in enumerate(rows):
            y = y_of(t + 2 + j * line)
            ax.text(x_label, y, label, ha="left", va="top", fontsize=6.0, color=FAINT, transform=tr)
            ax.text(x_text, y + 0.004, d[key], ha="left", va="top", fontsize=7.2, color=colour, transform=tr,
                    **font)
        ax.text(x_text, y_of(t + 2 + 3 * line + 1), verdict(d["gain"]), ha="left", va="top", fontsize=7.2,
                color=d["colour"], transform=tr)
        if i < len(decisions) - 1:
            ax.plot([x_label, 1], [y_of(t + block - 8)] * 2, color=RULE, lw=0.5, transform=tr, clip_on=False)


def panel_b(ax, name, task, agent, base, decisions):
    budget, lo = task.spec.budget, np.log10(FLOOR)
    cfg = CAMPAIGNS[name]
    best, b = base
    ax.grid(axis="y", lw=0.35, color="#EBEBEB", zorder=0)
    ax.plot(np.arange(1, len(b) + 1), b, color=GREY, lw=1.0, zorder=2)
    ax.text(*cfg["base_label"], f"{METHODS[best][0]}, best baseline", fontsize=6.5, color=GREY,
            ha="left", va="top")
    x = np.arange(1, budget + 1)
    for d in decisions:
        c, k = d["curve"], d["after"]
        ax.plot(x[k - 1:], c[k - 1:], color=d["colour"], lw=1.0, ls=(0, (3.2, 1.8)), zorder=3)
        ax.plot([budget], [c[-1]], "o", ms=3, color=d["colour"], zorder=4, clip_on=False)
        lx, ly = d.get("label_at", (budget - budget / 60, c[-1] + 0.25))
        ax.text(lx, ly, d["instead"], ha="right", va="bottom", fontsize=6.5, color=d["colour"])
    ax.plot(np.arange(1, len(agent) + 1), agent, color=AGENT, lw=1.7, zorder=5, solid_joinstyle="miter")
    ax.text(*cfg["agent_label"], "Agent", color=AGENT, fontsize=7.2, ha="right", va="center", **BOLD)
    for i, d in enumerate(decisions):
        badge(ax, d["after"], agent[d["after"] - 1], i + 1, d["colour"], size=10)
    top = np.ceil(max(agent.max(), b.max()))
    ax.set_xlim(0, budget)
    ax.set_ylim(lo - 0.5, top + 0.6)
    ax.set_yticks(np.arange(-8, top + 1, 2))
    ax.set_yticklabels([f"$10^{{{int(v)}}}$" for v in np.arange(-8, top + 1, 2)])
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Regret")
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
    fig = plt.figure(figsize=(WIDTH, 2.45))
    ax_a = fig.add_axes([0.035, 0.04, 0.44, 0.83])
    ax_b = fig.add_axes([0.585, 0.17, 0.405, 0.70])
    panel_a(fig, ax_a, a.campaign, decisions)
    panel_b(ax_b, a.campaign, task, agent, base, decisions)
    for ax, letter, title in ((ax_a, "a", "The task, and the agent's key decisions"),
                              (ax_b, "b", "Had the agent not made each decision")):
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
