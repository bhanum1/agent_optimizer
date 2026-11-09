# Agentoptimizer

An optimization assistant you run as a skill in Claude Code or Codex. You describe
what you want to optimize. The agent picks an optimizer and its settings for each
block of measurements, and hands you the points to measure. You measure them in the
lab, a simulation or a training run, and report the results. The agent then
decides the next block.

The repository has two parts:

- **[`skill/agentoptimizer/`](skill/agentoptimizer/)** is the product: the skill,
  its decision method, an optimizer library and the `aopt` command that keeps the
  campaign's records.
- **[`benchmark/`](benchmark/README.md)** is the research harness. It runs the same
  skill on benchmark tasks under a sandbox, with a program standing in for the user,
  and compares it with baseline optimizers.

## Install the skill

You need [uv](https://docs.astral.sh/uv/) 0.12 or later, and Claude Code or Codex.

```bash
skill/agentoptimizer/install.sh
```

This links the skill folder into `~/.claude/skills` and `~/.codex/skills`. Pass
`claude` or `codex` to install for only one of them, or `--uninstall` to remove the
links. It also builds the skill's Python environment, which includes torch and
botorch, so the first build takes a few minutes. Edits to the folder take effect
right away.

## Use it

In your project folder, ask the agent to optimize something, for example "help me
optimize the yield of this reaction", or name the agentoptimizer skill.

1. **Setup.** The agent interviews you about the objective, the variables (their types
   and ranges), constraints, noise, budget, earlier data, and what you already know.
   It writes `.agentoptimizer/problem.yaml` and `PROBLEM.md`, checks them with you,
   and imports your data.
2. **Each block.** The agent analyses the data, chooses an optimizer and its settings
   and decides how many points to measure. It records its reasoning in
   `decisions.md` and puts the first point(s) in `.agentoptimizer/data.csv`.
3. **Measuring.** Measure the pending point and tell the agent the value, or fill in
   the value column of `data.csv` and run `aopt suggest`. The optimizer takes the
   result and proposes the next point. Bayesian optimization refits on every
   result and gives one point at a time. A design block, such as Sobol points,
   lists all its points at once.
4. When the block is done, invoke the agent again. It reviews the data and decides
   the next block, until the budget is spent.

Variables can be `continuous` (optionally on a log scale or rounded to a
resolution), `integer`, `ordinal` or `categorical`. A finite list of allowed
configurations can be given as a candidates file. On noisy problems the agent may
ask for deliberate replicates. The format is in
[problem-format.md](skill/agentoptimizer/references/problem-format.md).

### The `aopt` command

The agent runs `aopt` for you, but you can run it yourself from the folder that holds
`.agentoptimizer/`. In these examples `aopt` stands for
`~/.claude/skills/agentoptimizer/aopt`, or `skill/agentoptimizer/aopt` in this
repository.

| Command | What it does |
|---|---|
| `aopt status` | Shows the budget used, the best point so far, the current block and the points to measure. |
| `aopt suggest` | Passes new results back to the block's optimizer and requests the next point(s). |
| `aopt init [--import data.csv]` | Creates `.agentoptimizer/` around a `problem.yaml`. |
| `aopt validate`, `aopt commit` | Checks the agent's proposal and records it as a new block. |
| `aopt encode`, `aopt decode` | Convert between a configuration and the optimizer's numeric point. |
| `aopt python script.py` | Runs Python with the optimizer runtime. |

The project folder:

```text
.agentoptimizer/
  problem.yaml PROBLEM.md      the problem definition and what you know about it
  data.csv                     every requested and measured point, in your units
  decisions.md                 the agent's reasoning, one entry per block
  agent/                       the agent's proposal and notes to its next call
  blocks/ state/ work/         each block's optimizer, saved optimizer state, analysis scripts
```

## Develop

```bash
uv run --project skill/agentoptimizer pytest
```

These tests drive whole blocks through `aopt` on synthetic problems, covering every
library optimizer, mixed variable types, reloads between measurements, replicates,
failed points and cancelled points.
[`AGENTS.md`](AGENTS.md) has the contributor rules.

## Layout

```text
skill/agentoptimizer/
  SKILL.md                     the skill: interview, setup, each invocation
  METHOD.md                    the decision method (also the benchmark agent's prompt)
  aopt                         the command; scripts/aoptcore/ is its Python core
  lib/                         optimizer library; host.py and check.py run and test optimizers
  references/                  problem format and variable encoding
  install.sh  pyproject.toml   install, and the optimizer runtime's dependencies
  tests/
benchmark/
  src/aobench/                 tasks, baselines, sandboxed runner
  data/                        benchmark inputs and the pipelines that produced them
  external/, runs/             git-ignored private data and outputs
```
