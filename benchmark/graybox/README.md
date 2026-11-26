# Gray-box results

Completed campaigns on 14 real tasks, with the agent run in two information
modes: **black box**, where it sees only anonymous variable columns, and
**gray box**, where it is also given a description of what each variable means.
These are the results behind Figure 4.

Imported from `Mahyar-rajabi94/cheapest_lunch` at `ebe7e4b`, which produced
them; it is a separate repository with its own task evaluators and its own copy
of the skill. Nothing here was run from this repository, and the tasks are not
reproducible from it. They are committed under the same exception as
`clean_runs/` and `figures/` (see [AGENTS.md](../../AGENTS.md)): a readable
summary of a finished run, not raw output.

## Arms

| Arm | Method in `summary.csv` |
|---|---|
| Baselines | `gp_bo`, `sobol_gp_bo`, `turbo`, `pool_random`, `batch_gp_qlogei`, `gp_qlogehvi`, `batch_random` |
| Agent, black box | `cheapest_lunch_opus_5_5_high` (`information_mode = anonymous`) |
| Agent, gray box | `cheapest_lunch_opus_5_5_high_domain_context` (`information_mode = domain-context`) |

Also present, and not part of the three-arm comparison:
`cheapest_lunch_opus_5_5_high_gray_box_rep_designer` (a fourth agent variant),
`cheapest_lunch_sonnet_5_5_medium_domain_context`, plain `cheapest_lunch`, and
`opus_5_5_high_prior_anonymous` / `opus_5_5_high_prior_domain_context`, which are
offline prior-ranking experiments carrying `evaluation.json` instead of a trace.

All three arms are present on nine tasks: `btch`, `mat_1`, `mat_2`, `mat_5`,
`mat_6`, `mix_1`, `rct_1`, `rct_2`, `sup`, at two or three seeds each.
`mat_3` is gray-box only, `mol_3` black-box only, and `mol_1`, `mol_2`, `ptr`
have baselines only. Budgets differ by task (100, 250 or 1000) and are recorded
per row; `ptr` used 1000 candidates here although another comparison used 5000.

## Layout

```text
summary.csv                                 one row per campaign, 301 rows
manifest.json                               which attempt each row selected
TASKS.md                                    what each task optimizes, and its units
DOMAIN_CONTEXT.md                           the 12 domain-context runs
tasks/<task>/<method>/seed_<n>/
  metadata.json                             task, method, seed, budget, model, effort
  trace.csv                                 the campaign, one row per evaluation
  evaluation.json                           offline prior runs, instead of trace.csv
  agent_artifacts/                          domain-context runs only
    decisions.md                            the agent's block-by-block reasoning
    domain-context.json                     the variable descriptions it was given
    final_agent/context.md                  its own notes at the end
```

Scalar traces carry `eval_idx`, `objective_value`, `best_objective`,
`simple_regret`, `block_or_batch` and `label`. Batch traces (`btch`, `ptr`)
carry `batch`, `position_in_batch`, `best_raw_after_batch` and
`regret_after_batch` instead; every row of a batch shares the post-batch
incumbent, so they are not revealed one at a time. The evaluator minimizes.

## Two things to know before plotting

**Regret is in task units and is not normalized across tasks.** Unlike Figure 2,
there is no ratio-to-best doing that work, so averaging across tasks needs a
normalization choice first.

**Final regret saturates.** These are pool tasks with a finite candidate set, and
on `btch`, `mat_2`, `mix_1`, `rct_1` and `rct_2` every arm reaches the pool
optimum exactly, so final regret is 0 and distinguishes nothing. What separates
the arms is *how fast* they get there — evaluations to reach the optimum, or
regret against fraction of budget.

## Pool labels

`pool/<task>.csv` holds one column: the task's label for every row of its candidate pool,
in the pool's own order, for `mix_1`, `sup`, `mat_5`, `rct_1` and `btch`. They exist for one
purpose, which is offline scoring: `aobench.fig4` takes a percentile of them to find what
counts as a top-1% candidate. 364 KB in all, and byte-identical to the last column of each
`data/<task>/main.csv` upstream.

No descriptors and no identities came with them. Those stay with the trusted evaluator, and
nothing here is anything an optimization agent may see.

## What was left behind

The source repository also holds 827 MB of agent reasoning
(`results/agent_reasoning/`, 24,201 transcript and prompt files), 304 MB of task
data including full-pool labels, and 70 MB of saved plots. None of it is here.
The per-call transcripts under each `agent_artifacts/` are also left out; the
decision journals that summarize them are kept.

The task data is deliberately excluded for a second reason: it holds the
descriptors and identities the trusted evaluator uses, which must never reach an
optimization agent. Only the label columns above came over, for scoring.
