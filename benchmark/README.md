# Benchmark

This folder runs the agentoptimizer skill on benchmark tasks and compares it with
baseline optimizers. The runner plays the user:

1. It writes the task as a `problem.yaml` with anonymous variables `x0..` and
   objective `y`. `PROBLEM.md` says the objective's identity is withheld, so the
   agent has no domain knowledge.
2. It calls the agent once per block, fresh, with the skill's
   [METHOD.md](../skill/agentoptimizer/METHOD.md) as its prompt. The interactive
   skill's interview is skipped. The agent writes a proposal and checks it with
   `aopt validate`.
3. It commits the block with the skill's core (`aoptcore`). Then it works through
   the block as a user would: the core asks the block's optimizer for a point, the
   runner evaluates the task, records the value in `data.csv` and passes it back.
4. It repeats until the budget is spent.

So the benchmark measures the same method, library and records as the shipped skill.
The agent never touches the objective. The runner drives either Codex CLI or Claude
Code CLI (`--agent codex|claude`); both use Codex's native sandbox for every agent
command and optimizer.

## Requirements

- macOS (checked) or Linux.
- [uv](https://docs.astral.sh/uv/) 0.12 or later.
- [Codex CLI](https://github.com/openai/codex), for **both** agents: it is the
  per-command sandbox. Authenticate it (`codex login`) only for `--agent codex`.
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code), only for
  `--agent claude`.

## Install

Run from the repository root. There are two environments, both built from lock
files:
- `.venv` is the evaluator: this package, the baselines, every objective backend,
  and the skill's core.
- `.agent-venv` is the agent runtime, built from the skill's own project: the
  optimizer libraries and the core. The runner refuses an agent runtime that contains
  this package or an objective backend.

```bash
uv sync --all-extras --locked
UV_PROJECT_ENVIRONMENT="$PWD/.agent-venv" uv sync --project skill/agentoptimizer --no-dev --locked
```

A bare `uv sync` (no `--all-extras`) removes the objective backends from `.venv`.

### Task data and backends

| Tasks | Needs |
| --- | --- |
| BBOB tasks | `ioh` (always installed) |
| `cec_*`, `noncont_rastrigin30`, `schaffer*` | `opfunu`, `setuptools<81` |
| `pymoo_g04`, `pymoo_g07` | `pymoo` |
| `constrained_*`, `noisy_rosenbrock_*`, `mixint_*` | `coco-experiment` |
| `ng_noisy_*` | `nevergrad` |
| `opv_pce`, `opv_pool`, `mof_pool` | files in `benchmark/data/` |
| `yahpo_svm`, `yahpo_xgboost` | `onnxruntime` plus surrogates in `benchmark/external/yahpo_data/` |
| `psa_v10` | a user-supplied evaluator at `benchmark/external/psa_v10/psa_v10_blackbox.py` (not present) |

`benchmark/external/` is git-ignored. Download the YAHPO surrogates with:

```bash
for s in rbv2_svm rbv2_xgboost; do for f in config_space.json encoding.json model.onnx; do curl -sfL --create-dirs -o benchmark/external/yahpo_data/$s/$f https://raw.githubusercontent.com/slds-lmu/yahpo_data/v1.0.2/$s/$f; done; done
```

Each folder in `benchmark/data/` has a README with its provenance and the pipeline
that built it.

## Run the agent

First check isolation. This makes no model call and no evaluations, and ends with
`Isolation passed`:

```bash
.venv/bin/python -m aobench.loop --task step_ellipsoid10 --budget 100 --check-only
```

Then run:

```bash
.venv/bin/python -m aobench.loop --task step_ellipsoid10 --budget 100
```

- **Defaults:** `--agent claude --model claude-opus-5-5 --effort high`.
- **`--budget`** counts objective evaluations.
- **`--seed`** seeds the agent's optimizers and any private observation noise.
- **`--context persistent`** (Claude only) keeps one session across calls instead of
  starting fresh each time.

Each invocation creates a new `benchmark/runs/run-<id>/`. Task keys are in
[the registry](src/aobench/tasks/registry.py).

### What the agent can access

Each call is a fresh CLI whose one tool, `run_command`, runs under Codex's sandbox:

- **Readable:** the run's workspace and `.agent-venv`. The workspace holds a copy of
  the skill folder and the project folder `.agentoptimizer/`: problem, `data.csv`,
  `decisions.md`, earlier blocks and state.
- **Writable:** `.agentoptimizer/work/`, `.agentoptimizer/agent/` and a temporary
  directory.
- **Denied:** the repository, the objective, `private/`, other runs, the network.

Optimizers run under a second policy that may write only `.agentoptimizer/state/`.
Preflight checks both policies, and that `skill/aopt` runs inside the sandbox, before
any model call.

### Outputs

```text
benchmark/runs/run-<id>/
  results/     trace.csv, meta.json, decisions.md (the baseline format)
  workspace/   what the agent saw: skill/ and .agentoptimizer/
  private/     journal.csv, blocks.json, per-call and per-block logs, audit.json, policy and manifest
```

A post-run audit checks four things: the budget, that `data.csv` agrees with the
private journal, that `decisions.md` holds one entry per committed block, and that
each block's rows match its proposal. A failed audit stops before `results/` is
written.

### How a run flows through the code

All under `src/aobench/loop/`:

1. [`main.py`](src/aobench/loop/main.py) parses the flags, builds the task from the
   registry and creates the run folder.
2. `Campaign.prepare` in [`campaign.py`](src/aobench/loop/campaign.py) does the setup:
   - copies the skill folder into the workspace;
   - writes `problem.yaml` and `PROBLEM.md` from the task, plus `candidates.csv` for
     an indexed pool and `prior.csv` if the task has one;
   - runs the core's `init`.
3. [`preflight.py`](src/aobench/loop/preflight.py) checks, with real sandboxed
   commands, that the agent cannot read the task, the repository or `private/`.
4. `Campaign.run` repeats until the budget is spent:
   - [`agents.py`](src/aobench/loop/agents.py) starts one Claude or Codex CLI call,
     with `METHOD.md` as the system prompt. Its only tool is `run_command`
     ([`tools.py`](src/aobench/loop/tools.py)), which runs through the sandbox
     ([`sandbox.py`](src/aobench/loop/sandbox.py)).
   - The core's `Project.commit` validates the agent's proposal and records it. A
     rejected proposal is noted in `decisions.md` and costs no evaluations.
   - `Campaign.run_block` opens a core `driver.Session` whose optimizer runs in
     `skill/host.py` inside the sandbox. The host only proposes points. The runner
     evaluates the task, records the value with the core, and appends it to the
     private journal ([`data.py`](src/aobench/loop/data.py)).
5. [`results.py`](src/aobench/loop/results.py) audits the run and writes `results/`.

## The axis study (`l18_01`..`l18_18`)

The 18 tasks `l18_01`..`l18_18`
([`tasks/synthetic.py`](src/aobench/tasks/synthetic.py)) form an L18 orthogonal
array over six axes, each with three levels:

- **modality:** unimodal, strong global structure, weak global structure;
- **added conditioning:** 1, 10³, 10⁶;
- **dimension:** 5, 10, 20;
- **noise:** additive, y = f + σ·N(0, 1), with σ = 0, 0.1× or 1× the gap of a point
  better than 90% of random points;
- **variable type:** continuous, 80% integer, 80% categorical;
- **effective dimension:** the share of the variables that matter (100%, 50%, 20%).

Each axis's level appears in 6 tasks, and every pair of levels across two axes
appears together.

Each task is a BBOB function from `ioh` with wrappers for the other axes. Which variables
are active and which are discrete is fixed per task. The seed selects the BBOB
instance, the category permutations, the conditioning rotation and the noise.
`f_opt` stays exact. Each
task's `suggested_budget` is 10d + 50.

## Add a task

A task is a class with a box, a known optimum (used only for scoring) and an objective.
All tasks minimize.

1. Subclass `Task` from [`tasks/base.py`](src/aobench/tasks/base.py) in a new file
   under `src/aobench/tasks/`. Set `name` (the results folder), `dim`, `bounds`
   (shape `(dim, 2)`) and `f_opt`, then implement `_f`:

   ```python
   import numpy as np
   from .base import Task

   class Quadratic(Task):
       def __init__(self, dim=4):
           self.name = f"quadratic_d{dim}"
           self.dim = dim
           self.bounds = np.array([[-5.0, 5.0]] * dim)
           self.f_opt = 0.0

       def _f(self, x):
           return float(np.sum((x - 1.0) ** 2))  # negate here to maximize
   ```

   - **Constrained:** set `n_constraints` and implement `_g` (feasible iff every value is ≤ 0).
   - **Noisy:** set `noisy = True` and implement `_f_true`, the noise-free value used for
     scoring. Override `reseed` if the noise has its own random generator.
   - The existing tasks in `tasks/` are short and show each case.

2. Add one line to `TASKS` in [`tasks/registry.py`](src/aobench/tasks/registry.py),
   with a comment saying what the task tests:

   ```python
   "quadratic4": lambda: Quadratic(4),
   ```

3. Check it. Both commands pick the new key up with no other change:

   ```bash
   .venv/bin/python -m aobench.baselines --task quadratic4 --optimizer cma_es --seed 0 --budget 40 --outdir benchmark/runs/try
   .venv/bin/python -m aobench.loop --task quadratic4 --budget 40 --check-only
   ```

Keep the task's file inside `tasks/`, because preflight checks that the agent cannot
read it. If the objective needs a new package, add it with `uv add` (or as an
extra), never to the skill's project.

### A task from a CSV of measured candidates

If you have a table of candidates with measured values, the finite-pool task
[`CSVPoolTask`](src/aobench/tasks/csv_pool.py) needs no new code. The file must have
the columns `id,x0,x1,...,y`. Each evaluation picks one unused row and reveals its `y`,
so the budget can be at most the number of rows. Register it:

```python
"my_pool": lambda: CSVPoolTask(DATA / "my_pool/data.csv", "my_pool", maximize=True),
```

- **`maximize=True`** negates `y`.
- **`prior=<csv with x0,...,y>`** gives the agent earlier measurements for free. They
  are copied into the workspace as `.agentoptimizer/prior.csv` and do not count
  against the budget.
- **Where tables go:** put private tables in `benchmark/external/`, which git
  ignores. `benchmark/data/` is committed.
- **Pool baselines** are `pool_random`, `gp_bo` and `turbo`.

## What the agent's strategy comes from

The skill folder defines the agent's behaviour: `METHOD.md` (its prompt) and `lib/`
(the optimizers it can name). Both are part of the experimental condition, so a
change to either changes what every later run measures. Commit the change and say why
in the commit message. Each run's `private/manifest.json` records the commit and any
uncommitted changes, so a run made from an edited tree can be told apart from one
made from a clean commit.

Runner-side rules are in `aoptcore`'s `Project.validate` (accepted proposal fields and
block lengths) and in `Campaign` (the per-call prompt; three failed calls before the
runner stops). Check a change with `--check-only`, then a short run such as
`--budget 40` on a cheap task.

## Baselines

One task, one optimizer, one seed:

```bash
.venv/bin/python -m aobench.baselines --task step_ellipsoid10 --optimizer cma_es --seed 0 --budget 100 --outdir benchmark/runs/baselines-100
```

Optimizers: `bobyqa`, `cma_es`, `cobyla`, `de`, `gp_bo`, `lbfgsb`, `nelder_mead`,
`random_search`, `smac`, `sobol`, `turbo`, `pool_random` (pool tasks only). Indexed
pools (`mof_pool`) use the exact-row `gp_bo`, `turbo` and `pool_random`. Each campaign
writes `<outdir>/<task_name>/<optimizer>/seed_<seed>/{trace.csv,meta.json}`.
