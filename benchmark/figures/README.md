# Figures

The benchmark's current figures, kept in git. Each folder has `regret` (mean log₁₀
regret per task, ±1 standard error) and `rank` (average rank over tasks, with each
task's rank behind it), as PDF and PNG.

| Folder | What | Seeds |
|---|---|---|
| `baselines/` | the eight baselines | 0–9 |
| `opus_high/` | the agent on Claude Opus 5.5, effort high, against the baselines | 0–4 |
| `sonnet_medium/` | the agent on Claude Sonnet 5.5, effort medium, against the baselines | 0–2 |

The agent figures use only the seeds the agent ran (`--seeds`), so every method is
compared on the same instances. To redraw them from the run folders:

```bash
.venv/bin/python -m aobench.figures --runs benchmark/runs/main --out benchmark/figures/baselines
.venv/bin/python -m aobench.figures --runs benchmark/runs/main benchmark/runs/opus_high --out benchmark/figures/opus_high --seeds 5
.venv/bin/python -m aobench.figures --runs benchmark/runs/main benchmark/runs/sonnet_medium --out benchmark/figures/sonnet_medium --seeds 3
```

`counterfactual_ill_conditioned_noisy` is a one-off study of Opus high on
`ill_conditioned_noisy`, seeds 0–5. At each block where the agent switched method or
restarted, the campaign was replayed to that point and the optimizer it was using was
run to the end of the budget instead. Each branch is a single noise draw, so
differences under about two decades may be luck. "(approx.)" marks branches after the
replay stopped matching the original run exactly. The script that made it is not in
the repository.

`fig2/` is Figure 2 of the paper, against seeds 0–4. The log₁₀ regret ratio is log₁₀ of
a method's regret divided by the best method's on the same instance (0 = best). (a) A
2-D slice of each task's landscape through the optimum of the seed-0 instance, as
observed, with insets on the two 1%-noise tasks showing that noise at the scale of a
finite-difference step;
(b) each optimizer's final ratio per task, coloured by size on panel a's scale, its
worst task labelled and a task it was best on, on every seed, drawn as a stub in the
optimum's colour and marked "best"; (c) with the agent (Opus high), the final ratio
averaged over tasks (mean ± s.e. over seeds); (d) the ratio against the fraction of the
budget spent, averaged over tasks and seeds. SMAC and Shiwa are left out: CMA-ES beats
SMAC on every task. To redraw:

```bash
.venv/bin/python -m aobench.fig2 --runs benchmark/runs/main --agent benchmark/runs/opus_high --seeds 5 --out benchmark/figures/fig2
```

`fig3/` is Figure 3 of the paper: one campaign, Opus high on `few_evals_10d` seed 1 (a
10-D sphere with 1% noise, 300 evaluations), chosen because the agent leads the best
baseline from start to end. (a) The task, and the agent's two key decisions as one turn
each of its loop: what it observed, what it inferred from that, and the optimizer it then
designed (from its rationales), with what the decision bought; (b) regret against evaluations for
the agent, the best baseline on that instance (Bandit), and one counterfactual per
decision: what would have run had the agent not made it. A counterfactual rewinds the
campaign to the decision, replaying its noise draws, and runs the agent's own previous
optimizer on to the end of the budget (`aobench.counterfactual`). The branches live in
`benchmark/runs/counterfactual`; the commands that make them are in the docstring of
`aobench/fig3.py`. To redraw:

```bash
.venv/bin/python -m aobench.fig3 --out benchmark/figures/fig3
```

`fig4/` is Figure 4 of the paper: the same agent on real discovery pools, with and without
the variable descriptions (black box and gray box). (a) The acceleration factor over
random search to a top-1% candidate, on the five scalar tasks of the gray-box campaign,
with each arm's geometric mean as a dashed rule across the panel. The baseline bar is the
best classical optimizer the task carries, picked after the fact, so it is a generous
comparator. `rct_2` is left out: it is multiobjective, so there is no single column to
take a percentile of. The target has to be a demanding one — at top 5% random search
already arrives in 11 to 34 experiments on most of these pools, and every method's initial
design finds it by luck (`--top` shows this). (b, c) Two of those campaigns in the form of
Figure 3, sharing its card panel (`aobench.fig3.cards`): mixture discovery (`mix_1` seed 1)
and MOF discovery (`mat_5` seed 1), the agent's key decisions quoted from each run's
`decisions.md`, against what they bought. The results they read live in
`benchmark/graybox/`, and the pool label columns used for the top-1% cut are in
`benchmark/graybox/pool/` (offline scoring only; no agent sees them). To redraw:

```bash
.venv/bin/python -m aobench.fig4 --out benchmark/figures/fig4
```
