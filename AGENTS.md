# Contributor guide

Instructions for people and coding assistants working on this repository. The
skill's own prompts are different files:
[skill/agentoptimizer/SKILL.md](skill/agentoptimizer/SKILL.md) (the interactive skill)
and [METHOD.md](skill/agentoptimizer/METHOD.md) (the decision method). See
[README.md](README.md) for the skill and [benchmark/README.md](benchmark/README.md)
for the benchmark.

## Rules

- **Two projects.** `skill/agentoptimizer/` is a standalone uv project (the skill's
  runtime and its core, `aoptcore`). The root project is the benchmark (`aobench`,
  in `benchmark/src/`). It depends on the skill as an editable path dependency. The
  skill never imports `aobench`.
- **Dependencies** live in `pyproject.toml` and `uv.lock` files. Change them with
  `uv add` / `uv lock` (`--project skill/agentoptimizer` for the skill), never by
  installing into an environment by hand. Two environments:
  - `.venv` is the evaluator.
  - `.agent-venv` is the benchmark's agent runtime, built from the skill project
    alone. Never install `aobench` or any objective backend into it.

  The skill's `[tool.uv] constraint-dependencies` and the root's must agree: they
  pin the optimizer libraries.
- **The skill folder is an interface.** It is copied as-is into each benchmark
  sandbox and linked into users' skills directories, so put nothing there a user or
  the agent should not see. Keep `lib/` importable on its own (flat imports such as
  `from _base import Base`), with no import of `aoptcore` or `aobench`. `aoptcore`
  needs only the skill project's dependencies.
- **The skill is an experimental condition.** Changing `METHOD.md`, `lib/` or the
  core's validation and driver changes what every later benchmark run measures.
  Record why in the commit message. `SKILL.md` affects only interactive use.
- **Task identities stay private.** Nothing from `benchmark/src/aobench/tasks/` may
  reach a sandbox. The benchmark's `PROBLEM.md` carries no domain knowledge.
- **Test the core** with `uv run --project skill/agentoptimizer pytest` after changing
  `aoptcore` or `lib/`.
- **No results in git.** Campaign outputs go to `benchmark/runs/` (ignored).
