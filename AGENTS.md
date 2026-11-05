# Contributor guide

Instructions for people and coding assistants working on this repository.
The prompt given to the optimization agent is a different file:
[src/agentoptimizer/loop/SKILL.md](src/agentoptimizer/loop/SKILL.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/agentoptimizer/` | The package: `loop/` (the agent loop: runner, sandbox, optimizer host, `SKILL.md`, audit; `lib/` is the agent's optimizer library), `tasks/` (objectives and `TASKS` registry), `optimizers/` (ask/tell wrappers the baselines run) |
| `scripts/` | `run_loop.py` (one agent campaign); `run_campaign.py` and `run_sweep.py` (baselines); `figures/` for paper figures; `slurm/` for cluster sweeps |
| `data/` | Task inputs (MOF pool, OPV emulator, OPV pool) and the pipelines that produced them, including the lab-mixture builders whose data stays in `external/` |
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

- **The agent's workspace is an interface.** The runner copies `SKILL.md`,
  `check.py`, `host.py` and `lib/` from `loop/` into each sandbox. Keep `lib/`
  importable on its own (flat imports such as `from _base import Base`), limited
  to the `agent` dependency group, and free of task identities.
- **The skill is an experimental condition.** Changing `loop/SKILL.md` or the
  library changes what every later run measures; record why in the commit
  message.
- **Task identities stay private.** `docs/TASKS.md` and `src/agentoptimizer/tasks/`
  name the functions; nothing from them may reach a sandbox.
- **No results in git.** Campaign outputs go to `runs/` (ignored) and are
  released separately; see `docs/plan/04-reproducibility-and-release.md`.
- Plans and open decisions live in `docs/plan/`; update them when a decision
  is made.
