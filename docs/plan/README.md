# Manuscript plan

Working plan for turning this repository into the code release for the paper.
These documents hold decisions and their reasons; code arrives in the PR that
implements each piece, not as placeholders ahead of time.

This repository was seeded from `bhanum1/Cheapest_opt_lunch` at tag
`pre-manuscript-snapshot` (`40c94b98`). That repository keeps the full
development history and every result produced before the cutover.

## Documents

| Doc | Covers |
| --- | --- |
| [01-repo-structure.md](01-repo-structure.md) | Target layout, migration from the seed, cleanup list |
| [02-dynamic-selection-baselines.md](02-dynamic-selection-baselines.md) | Non-LLM methods that select or switch optimizers |
| [03-llm-optimizer-comparisons.md](03-llm-optimizer-comparisons.md) | LLM-based optimizers and benchmarks we compare against |
| [04-reproducibility-and-release.md](04-reproducibility-and-release.md) | Environments, reruns, data release, license, citation |
| [05-advisor-mode.md](05-advisor-mode.md) | Real use: propose the next block from existing data, as a script |

## Claims → evidence

Draft claims, to be confirmed with co-authors. Each row names the experiment
that supports it and the figure or table that shows it.

| # | Claim (draft) | Experiment | Figure / table | Status |
| --- | --- | --- | --- | --- |
| C1 | An agent that selects and switches optimizers block by block matches or beats the single best solver (SBS) across a diverse suite and closes part of the gap to the virtual best solver (VBS). | 30-task suite, 2,000 evaluations | SBS/VBS plot, average-rank plot | Runs exist in the old repo; rerun from this repo planned (04) |
| C2 | The gain comes from adaptive selection, not only from picking a good single method up front. | Agent vs. non-LLM dynamic selectors (02) | Rank table incl. selectors | Not started |
| C3 | An agent that decides per block is competitive with LLM optimizers that act on every evaluation, at far fewer LLM calls. | Comparisons in both directions (03) | Score vs. LLM-cost plot | Not started |
| C4 | The behaviour carries over to real scientific tasks with small budgets. | OPV and MOF pools, 100 evaluations | Materials comparison figure | Runs exist in the old repo |
| C5 | The protocol keeps the agent blind to the objective and enforces the budget. | Sandbox preflight, integrity tests | Methods section, supplementary | Implemented; Linux path needs validation |
| C6 | The same policy works as an advisor that decides each block from a fresh context and proposes it as a batch. | Advisor replay (05) | Advisor vs. benchmark agent and batch baselines | Not started |

## Workstream status

| Workstream | Owner | Status | Next step |
| --- | --- | --- | --- |
| Repo restructure | | Layout done | Decide `psa_v10` and figure inputs (01) |
| Dynamic-selection baselines | | Planned | Confirm method list and fairness rules in 02 |
| LLM-optimizer comparisons | | Planned | Check code availability for each method in 03 |
| Reproducibility and release | | Planned | Choose data host and license in 04 |
| Advisor mode (`/opt-run`) | | Phases 1–3 done; replay runner works end to end (phase 4) | Fixed-q and no-summary flags; first C6 runs on OPV and MOF pools (05) |

## Open decisions

Collected from the documents below; resolve here and record the outcome in
the relevant document.

1. Ownership and credit: confirm with co-authors that the release lives at
   `bhanum1/agentoptimizer`, or move it to a lab organization.
2. ~~Flatten `main/`~~ Done (01). Still open there: `psa_v10`, data pipelines, figure inputs.
3. Code and data licenses (04).
4. Data host for results (04).
5. Final method lists for 02 and 03.
6. Advisor mode (05): domain context in replay.
