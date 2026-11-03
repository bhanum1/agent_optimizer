# Contributor guide

Instructions for people and coding assistants working on this repository.
The prompt given to the optimization agent in experiments is a different file:
[prompts/optimization_agent.md](prompts/optimization_agent.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/agentoptimizer/` | The package: `agent_harness/` (client the agent calls), `blind_runner/` (isolated runner, advisor replay, evaluator, preflight), `optimizers/` (ask/tell wrappers), `tasks/` (objectives and `TASKS` registry), `advisor/` (advisor mode: campaign folders for real experiments), `batch/` (stateless batch proposers from data: space filling, qLogNEI, trust region, CMA-ES, DE) |
| `scripts/` | `run_agent.py`, `run_campaign.py`, `run_sweep.py`; `run_replay.py` (advisor replay) and `run_batch_baselines.py` (its batch baselines); `agent_runs.py` (launch, wait, status); `opt_run.py` (advisor mode); `figures/` for paper figures; `slurm/` for cluster sweeps |
| `.claude/skills/` | `/optimize` (launch benchmark runs and report on them) and `/opt-run` (advisor mode: propose the next block of a real campaign) |
| `examples/` | Advisor-mode example campaigns with simulated labs |
| `prompts/` | The benchmark agent's prompt (`optimization_agent.md`), the advisor-mode policy (`advisor.md`) and the replay advisor's procedure (`advisor_headless.md`) |
| `data/` | Task inputs (MOF pool, OPV emulator, OPV pool) and the pipelines that produced them |
| `docs/` | Task catalogue, adding tasks, figure rules, and `plan/` for the manuscript |
| `tests/` | Unit tests; no model calls, no real objectives |

## Setup and checks

```bash
uv sync --all-extras --locked
UV_PROJECT_ENVIRONMENT=.agent-venv uv sync --only-group agent --locked
.venv/bin/python -m unittest discover -s tests
```

Dependencies live in `pyproject.toml` and `uv.lock`; change them with `uv add` /
`uv lock`, never by installing into an environment by hand. `.venv` is the
evaluator; `.agent-venv` is the agent runtime, built from the `agent` group
alone. Never install this package or any objective backend into it; the runner
refuses to start if it finds one. `[tool.uv] constraint-dependencies` pins the
versions behind the published runs: changing them changes the experimental
conditions, so say why in the commit message.

## Rules

- **The agent's workspace is an interface.** The runner copies
  `agent_harness/harness.py` and `optimizers/` into each sandbox as the package
  `src`. Keep imports inside those modules relative, and keep them free of task
  identities, so the copies work and reveal nothing.
- **The agent prompt is an experimental condition.** Changing
  `prompts/optimization_agent.md` changes what every later run measures; record
  why in the commit message. `prompts/advisor.md` carries the same block-wise
  policy into advisor mode; keep the two in step. `prompts/advisor_headless.md`
  is the replay advisor's per-block procedure and follows the `/opt-run`
  skill's steps; keep those in step too.
- **Task identities stay private.** `docs/TASKS.md` and `src/agentoptimizer/tasks/`
  name the functions; nothing from them may reach a sandbox.
- **No results in git.** Campaign outputs go to `runs/` (ignored) and are
  released separately; see `docs/plan/04-reproducibility-and-release.md`.
- Plans and open decisions live in `docs/plan/`; update them when a decision
  is made.
