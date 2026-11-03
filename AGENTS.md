# Contributor guide

Instructions for people and coding assistants working on this repository.
The policy given to the optimization agent is a different file:
[prompts/advisor.md](prompts/advisor.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/agentoptimizer/` | The package: `advisor/` (campaign folders: problem, data, summary, blocks and rounds), `batch/` (stateless batch proposers from data: space filling, qLogNEI, trust region, CMA-ES, DE), `blind_runner/` (the blind loop: headless advisor, sandbox, preflight, private evaluator), `tasks/` (objectives and `TASKS` registry), `optimizers/` (ask/tell wrappers the baselines run) |
| `scripts/` | `run_agent.py` (one blind campaign), `agent_runs.py` (launch, wait, status), `opt_run.py` (campaign CLI); `run_campaign.py`, `run_sweep.py` and `run_batch_baselines.py` (baselines); `figures/` for paper figures; `slurm/` for cluster sweeps |
| `.claude/skills/` | `/optimize` (launch blind runs and report on them) and `/opt-run` (propose the next block or round of a real campaign) |
| `examples/` | Example campaigns with simulated labs |
| `prompts/` | The advisor policy (`advisor.md`) and the headless advisor's per-block procedure (`advisor_headless.md`) |
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

- **The agent's workspace is an interface.** The runner copies `advisor/` and
  `batch/` into each sandbox as `lib/agentoptimizer`. Keep imports inside them
  relative, limited to the `agent` dependency group, and free of task
  identities, so the copies work and reveal nothing.
- **The prompts are an experimental condition.** Changing `prompts/advisor.md`
  or `prompts/advisor_headless.md` changes what every later run measures;
  record why in the commit message. `advisor_headless.md` follows the
  `/opt-run` skill's steps; keep the two in step.
- **Task identities stay private.** `docs/TASKS.md` and `src/agentoptimizer/tasks/`
  name the functions; nothing from them may reach a sandbox.
- **No results in git.** Campaign outputs go to `runs/` (ignored) and are
  released separately; see `docs/plan/04-reproducibility-and-release.md`.
- Plans and open decisions live in `docs/plan/`; update them when a decision
  is made.
