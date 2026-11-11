# Contributor guide

Instructions for people and coding assistants working on this repository. The
skill's own prompts are different files:
[skill/agentoptimizer/SKILL.md](skill/agentoptimizer/SKILL.md) (the interactive skill)
and [METHOD.md](skill/agentoptimizer/METHOD.md) (the decision method). See
[README.md](README.md) for the skill and [benchmark/README.md](benchmark/README.md)
for the benchmark.

## Rules

- **Two independent projects.** `skill/agentoptimizer/` is a standalone uv project
  (the skill's runtime and its core, `aoptcore`). The root project is the benchmark
  (`aobench`, in `benchmark/src/`): six baseline optimizers on a few BBOB tasks.
  Neither imports the other.
- **Dependencies** live in `pyproject.toml` and `uv.lock` files. Change them with
  `uv add` / `uv lock` (`--project skill/agentoptimizer` for the skill), never by
  installing into an environment by hand. The optimizer package versions pinned in
  the two projects' `constraint-dependencies` should agree.
- **The skill folder is an interface.** It is linked into users' skills
  directories, so put nothing there a user or the agent should not see. There is no
  optimizer library: the agent writes every optimizer itself, in the project's
  `work/`.
- **The benchmark stays small.** Its one purpose is to show that every solver fails
  somewhere: each baseline should have a task where it is good and one where it is
  bad. Add a task only when it shows something the others do not.
- **Test the core** with `uv run --project skill/agentoptimizer pytest` after changing
  `aoptcore`, `host.py` or `check.py`.
- **No results in git.** Benchmark outputs go to `benchmark/runs/` (ignored).
