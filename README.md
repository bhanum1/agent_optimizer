# Agentoptimizer

An optimization assistant you run as a skill in Claude Code or Codex. You describe
what you want to optimize. For each block of measurements the agent decides what to
do: it writes an optimizer and picks its settings, or it lists the points to measure
itself. You measure them (in the lab, a simulation or a training run) and report the
results. Then the agent decides the next block.

The repository has two independent parts:

| Part | What it is |
|---|---|
| [`skill/`](skill/) | The skill. [`SKILL.md`](skill/SKILL.md) is the **interview**: it sets up a campaign with the user and runs it. [`METHOD.md`](skill/METHOD.md) is the **method**: how the agent decides each block, by writing an optimizer or by listing points. The `aopt` command keeps the campaign's records and runs the agent's optimizers. |
| [`benchmark/`](benchmark/README.md) | The **harness**, which runs the skill on a black-box task with the harness as the user: it builds a sandbox, writes the starting files and calls the agent once per block. Also eight baseline optimizers on BBOB tasks, for comparison. |

## Install the skill

You need [uv](https://docs.astral.sh/uv/) 0.12 or later, and Claude Code or Codex.

```bash
skill/install.sh
```

This links the skill folder into `~/.claude/skills` and `~/.codex/skills`. Pass
`claude` or `codex` to install it for only one of them, or `--uninstall` to remove the
links. It also builds the skill's Python environment, which includes torch and
botorch, so the first build takes a few minutes. Edits to the folder take effect
right away.

## Use it

In your project folder, ask the agent to optimize something, for example "help me
optimize the yield of this reaction", or name the agentoptimizer skill.

1. **Interview.** The agent asks about the objective, the variables (their types and
   ranges), constraints, noise, budget, earlier data and what you already know. It
   writes `.agentoptimizer/problem.yaml` and `PROBLEM.md`, checks them with you and
   imports your data.
2. **Each block.** The agent analyses the data and makes a proposal of one of two
   kinds: an optimizer it writes, with its settings and the number of points, or a
   list of points it chose itself. It records its reasoning in `decisions.md` and
   puts the first point(s) in `.agentoptimizer/data.csv`.
3. **Measuring.** Measure the pending points and tell the agent the values, or fill in
   the value column of `data.csv` and run `aopt suggest`. An adaptive optimizer, such
   as Bayesian optimization, takes each result and proposes the next point. A design,
   or a list of points, is requested all at once.
4. When the block is done, invoke the agent again. It reviews the data and decides
   the next block, until the budget is spent.

Variables can be `continuous` (optionally on a log scale or rounded to a
resolution), `integer`, `ordinal` or `categorical`. A finite list of allowed
configurations can be given as a candidates file. On noisy problems the agent may
ask for deliberate replicates. The format is in
[problem-format.md](skill/references/problem-format.md).

### The `aopt` command

The agent runs `aopt` for you, but you can run it yourself from the folder that holds
`.agentoptimizer/`. Here `aopt` stands for `~/.claude/skills/agentoptimizer/aopt`, or
`skill/aopt` in this repository.

| Command | What it does |
|---|---|
| `aopt status` | Shows the budget used, the best point so far, the current block and the points to measure. |
| `aopt suggest` | Passes new results back to the block's optimizer and requests the next point(s). |
| `aopt suggest --serve` | The same for a whole block in one process, keeping the optimizer in memory: after each request, fill in the values and send a line on stdin. For scripted measurements. |
| `aopt init [--import data.csv]` | Creates `.agentoptimizer/` around a `problem.yaml`. |
| `aopt validate`, `aopt commit` | Check the agent's proposal, and record it as a new block. |
| `aopt encode`, `aopt decode` | Convert between a configuration and the optimizer's numeric point. |
| `aopt python script.py` | Runs Python with the optimizer runtime. |

The project folder:

```text
.agentoptimizer/
  problem.yaml PROBLEM.md      the problem definition and what you know about it
  data.csv                     every requested and measured point, in your units
  decisions.md                 the agent's reasoning, one entry per block
  agent/                       the agent's proposal and its notes to its next call
  blocks/ state/ work/         each block's optimizer, saved optimizer state, the agent's scripts
```

## Run the agent on a benchmark task

```bash
uv sync --locked
.venv/bin/python -m aobench.agent --out benchmark/runs/agent --tasks gallagher --seeds 1
```

The agent runs in a bubblewrap sandbox, so this needs Linux with `bwrap` and
`claude` installed. `--agent fake` swaps in a scripted agent and runs anywhere. See
[benchmark/README.md](benchmark/README.md).

## Layout

```text
skill/                           the skill (linked into the user's skills folder)
  SKILL.md                       the interview, and each invocation
  METHOD.md                      the decision method: write an optimizer or list points
  aopt                           the command; scripts/aoptcore/ is its Python core
  host.py, check.py              run, and test offline, the agent's optimizers
  references/                    the problem format and the variable encoding
  install.sh, pyproject.toml     install, and the optimizer runtime's dependencies
  tests/                         core tests, with small test optimizers
benchmark/
  src/aobench/
    tasks.py                     BBOB tasks, some noisy
    agent/                       the harness: sandbox, starting files, the agent loop
    baselines/                   eight baseline optimizers
    run.py                       runs the baselines and writes the report
  runs/                          outputs (git-ignored)
```

## Develop

```bash
uv run --project skill pytest
```

These tests drive whole blocks through `aopt` on synthetic problems: mixed variable
types, listed points, reloads between measurements, replicates, and failed and
cancelled points. [`AGENTS.md`](AGENTS.md) has the contributor rules.
