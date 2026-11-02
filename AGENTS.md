# Contributor guide

Instructions for people and coding assistants working on this repository.
The prompt given to the optimization agent in experiments is a different file:
[prompts/optimization_agent.md](prompts/optimization_agent.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/agentoptimizer/` | The package: `agent_harness/` (client the agent calls), `blind_runner/` (isolated runner, evaluator, preflight), `optimizers/` (ask/tell wrappers), `tasks/` (objectives and `TASKS` registry) |
| `scripts/` | `run_agent.py`, `run_campaign.py`, `run_sweep.py`; `figures/` for paper figures; `slurm/` for cluster sweeps |
| `prompts/` | The optimization agent's prompt |
| `data/` | Task inputs (MOF pool, OPV emulator, OPV pool) and the pipelines that produced them |
| `docs/` | Task catalogue, adding tasks, figure rules, and `plan/` for the manuscript |
| `tests/` | Unit tests; no model calls, no real objectives |

## Setup and checks

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install -e ".[figures]"
.venv/bin/python -m unittest discover -s tests
```

The agent runtime is a second environment (`.agent-venv`, from
`requirements-agent.txt`). Never install this package or any objective backend
into it; the runner refuses to start if it finds one.

## Rules

- **The agent's workspace is an interface.** The runner copies
  `agent_harness/harness.py` and `optimizers/` into each sandbox as the package
  `src`. Keep imports inside those modules relative, and keep them free of task
  identities, so the copies work and reveal nothing.
- **The agent prompt is an experimental condition.** Changing
  `prompts/optimization_agent.md` changes what every later run measures; record
  why in the commit message.
- **Task identities stay private.** `docs/TASKS.md` and `src/agentoptimizer/tasks/`
  name the functions; nothing from them may reach a sandbox.
- **No results in git.** Campaign outputs go to `runs/` (ignored) and are
  released separately; see `docs/plan/04-reproducibility-and-release.md`.
- Plans and open decisions live in `docs/plan/`; update them when a decision
  is made.
