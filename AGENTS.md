# Contributor guide

Instructions for people and coding assistants working on this repository.
The prompt given to the optimization agent is a different file:
[src/agentoptimizer/loop/SKILL.md](src/agentoptimizer/loop/SKILL.md).
See [README.md](README.md) for layout, install and usage.

## Rules

- **Dependencies** live in `pyproject.toml` and `uv.lock`; change them with
  `uv add` / `uv lock`, never by installing into an environment by hand. `.venv`
  is the evaluator; `.agent-venv` is the agent runtime, built from the `agent`
  group alone. Never install this package or any objective backend into it.
  `[tool.uv] constraint-dependencies` pins the versions behind the published
  runs: changing them changes the experimental conditions, so say why in the
  commit message.
- **The agent's workspace is an interface.** The runner copies `SKILL.md`,
  `check.py`, `host.py` and `lib/` from `loop/` into each sandbox. Keep `lib/`
  importable on its own (flat imports such as `from _base import Base`), limited
  to the `agent` dependency group, and free of task identities.
- **The skill is an experimental condition.** Changing `loop/SKILL.md` or the
  library changes what every later run measures; record why in the commit
  message.
- **Task identities stay private.** Nothing from `src/agentoptimizer/tasks/`
  may reach a sandbox.
- **No results in git.** Campaign outputs go to `runs/` (ignored).
