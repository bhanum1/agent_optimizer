# Agentoptimizer

An LLM chooses optimization methods, hyperparameters, and evaluation block lengths
for a black-box problem, one block at a time. A Python loop drives it:

1. Call the agent once, fresh, with [SKILL.md](src/agentoptimizer/loop/workspace/SKILL.md) as its prompt.
   It analyzes the data so far and writes a proposal: an optimizer (from the
   [library](src/agentoptimizer/loop/workspace/lib/README.md) or its own), its settings, and how many
   points to evaluate.
2. Validate the proposal and append the agent's decision to `decisions.md`.
3. Run that optimizer in a sandbox for exactly that many points. The runner
   evaluates every point itself and appends it to `data.csv`.
4. Repeat until the budget is spent.

The agent never touches the objective. The runner drives either Codex CLI or
Claude Code CLI (`--agent codex|claude`); both use Codex's native sandbox for every
agent command and optimizer.

Earlier results are in the source repository at tag
[`pre-manuscript-snapshot`](https://github.com/bhanum1/Cheapest_opt_lunch/tree/pre-manuscript-snapshot).

## Requirements

- macOS (checked) or Linux.
- [uv](https://docs.astral.sh/uv/) 0.12 or later.
- [Codex CLI](https://github.com/openai/codex), for **both** agents: it is the
  per-command sandbox. Authenticate it (`codex login`) only for `--agent codex`.
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code), only for
  `--agent claude`.

## Install

Two environments, both built from `uv.lock`. `.venv` is the evaluator: this
package, the baselines and every objective backend. `.agent-venv` is the agent
runtime: optimizer libraries only. The runner refuses an agent runtime that
contains this package or an objective backend.

```bash
uv sync --all-extras --locked
UV_PROJECT_ENVIRONMENT=.agent-venv uv sync --only-group agent --locked
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
| `opv_pce`, `opv_pool`, `mof_pool` | files in `data/` |
| `yahpo_svm`, `yahpo_xgboost` | `onnxruntime` plus surrogates in `external/yahpo_data/` |
| `psa_v10` | a user-supplied evaluator at `external/psa_v10/psa_v10_blackbox.py` (not present) |

`external/` is git-ignored. Download the YAHPO surrogates with:

```bash
for s in rbv2_svm rbv2_xgboost; do for f in config_space.json encoding.json model.onnx; do curl -sfL --create-dirs -o external/yahpo_data/$s/$f https://raw.githubusercontent.com/slds-lmu/yahpo_data/v1.0.2/$s/$f; done; done
```

Each folder in `data/` has a README with its provenance and the pipeline that built it.

## Run the agent

Check isolation first (no model call, no evaluations; ends with `Isolation passed`):

```bash
.venv/bin/python -m agentoptimizer.loop --task step_ellipsoid10 --budget 100 --check-only
```

Then run:

```bash
.venv/bin/python -m agentoptimizer.loop --task step_ellipsoid10 --budget 100
```

Defaults are `--agent claude --model claude-opus-5-5 --effort high`. `--budget` counts
objective evaluations. `--seed` seeds the agent's optimizers and any private
observation noise. Each invocation creates a new `runs/run-<id>/`. Task keys are in
[the registry](src/agentoptimizer/tasks/registry.py).

### What the agent can access

Each call is a fresh CLI whose one tool, `run_command`, runs under Codex's sandbox:

- **Readable:** the run's workspace (problem statement, `space.json`, `data.csv`,
  `decisions.md`, the library, earlier blocks and state) and `.agent-venv`.
- **Writable:** `work/`, `agent/` and a temporary directory.
- **Denied:** the repository, the objective, `private/`, other runs, the network.

Optimizers run under a second policy that may write only `state/`. Preflight checks
both policies before any model call.

### Outputs

```text
runs/run-<id>/
  results/     trace.csv, meta.json, decisions.md (the baseline format)
  workspace/   what the agent saw
  private/     journal.csv, blocks.json, per-call logs, audit.json, policy and manifest
```

A post-run audit checks budget, data integrity and that each block followed its
decision; a failed audit stops before `results/` is written.

### How a run flows through the code

All under `src/agentoptimizer/loop/`:

1. [`main.py`](src/agentoptimizer/loop/main.py) parses the flags, builds the task from the
   registry and creates `runs/run-<id>/`.
2. `Campaign.prepare` in [`campaign.py`](src/agentoptimizer/loop/campaign.py) copies
   `workspace/` into the run, writes `PROBLEM.md` and `space.json` (the only description of
   the task the agent gets) and an empty `decisions.md`.
3. [`preflight.py`](src/agentoptimizer/loop/preflight.py) checks, with real sandboxed
   commands, that the agent cannot read the task, the repository or `private/`.
4. `Campaign.run` repeats until the budget is spent:
   - [`agents.py`](src/agentoptimizer/loop/agents.py) starts one Claude or Codex CLI call
     with `SKILL.md` as the system prompt. Its only tool is `run_command`
     ([`tools.py`](src/agentoptimizer/loop/tools.py)), which runs through the sandbox
     ([`sandbox.py`](src/agentoptimizer/loop/sandbox.py)).
   - The agent writes `agent/proposal.json` and `agent/decision.md`. `Campaign.validate`
     checks the proposal; a rejected one is noted in `decisions.md` and costs no evaluations.
   - [`block.py`](src/agentoptimizer/loop/block.py) runs the chosen optimizer through
     `workspace/host.py` in the sandbox. The host only proposes points. The runner calls
     `task.evaluate` itself and appends each result to `data.csv` through
     [`data.py`](src/agentoptimizer/loop/data.py).
5. [`results.py`](src/agentoptimizer/loop/results.py) audits the run and writes `results/`.

Propose mode (`--data`, below) replaces the task with
[`user_data.py`](src/agentoptimizer/loop/user_data.py). It stops after one call:
`Campaign.propose` validates the proposal and returns it without running the block.

## Add a task

A task is a class with a box, a known optimum (used only for scoring) and an objective.
All tasks minimize.

1. Subclass `Task` from [`tasks/base.py`](src/agentoptimizer/tasks/base.py) in a new file
   under `src/agentoptimizer/tasks/`. Set `name` (the results folder), `dim`, `bounds`
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

2. Add one line to `TASKS` in [`tasks/registry.py`](src/agentoptimizer/tasks/registry.py),
   with a comment saying what the task tests:

   ```python
   "quadratic4": lambda: Quadratic(4),
   ```

3. Check it. Both commands pick the new key up with no other change:

   ```bash
   .venv/bin/python -m agentoptimizer.baselines --task quadratic4 --optimizer cma_es --seed 0 --budget 40 --outdir runs/try
   .venv/bin/python -m agentoptimizer.loop --task quadratic4 --budget 40 --check-only
   ```

Keep the task's file inside `tasks/`: preflight checks that the agent cannot read it. If the
objective needs a new package, add it with `uv add` (or as an extra), never to the `agent`
group.

### A task from a CSV of measured candidates

If you have a table of candidates with measured values, the finite-pool task
[`CSVPoolTask`](src/agentoptimizer/tasks/csv_pool.py) needs no new code. The file must have
the columns `id,x0,x1,...,y`. Each evaluation picks one unused row and reveals its `y`, so
the budget can be at most the number of rows. Register it:

```python
"my_pool": lambda: CSVPoolTask(DATA / "my_pool/data.csv", "my_pool", maximize=True),
```

`maximize=True` negates `y`. Pass `prior=<csv with x0,...,y>` to give the agent earlier
measurements for free: they are copied into the workspace as `prior.csv` and do not count
against the budget. Put private tables in `external/`, which git ignores; `data/` is
committed. Pool baselines are `pool_random`, `gp_bo` and `turbo`.

## Propose one block from your own CSV

For an experiment you run yourself, propose mode makes one skill call on your data and writes
out the block the agent chose. Nothing is evaluated, and the block is not run: you run it.

```bash
.venv/bin/python -m agentoptimizer.loop --data my_data.csv --bounds=-5,5 --remaining 10
```

- **`--data`:** a CSV with columns `x0,x1,...,y`. `y` is minimized, so negate a value you
  want to maximize. Every row becomes the run's `data.csv`.
- **`--bounds`:** either `lo,hi` for every variable, or a JSON list of `[lo, hi]` pairs, given
  inline or as a file. Write it with `=` when it starts with a minus sign.
- **`--remaining`:** how many evaluations you have left. The agent chooses the block length,
  up to that number.

The run uses the same sandbox, preflight and flags (`--model`, `--effort`, `--seed`) as a full
campaign. It writes to `results/`:
- `proposal.json`: the block. It names the optimizer script, its `args`, the number of
  evaluations `n_points`, and `state` and `warm_start`.
- The optimizer script itself, as the agent left it. This is either a library file or one the
  agent wrote in `work/`.
- `decision.md`: the agent's reasoning for the block.
- `context.md`: the notes it saved for its next call.

The script defines the `Optimizer` class described in
[`host.py`](src/agentoptimizer/loop/workspace/host.py), with `ask`, `tell`, `save` and `load`.
- **Running the block:** construct it with the space and `args`. If `warm_start` is true, tell
  it every row of your data first. Then alternate `ask(1)` and `tell` for `n_points`
  evaluations.
- **The next call:** append your measurements to the CSV and run propose mode again. Each run
  starts from all the data, so optimizer state does not carry over.

### Without the agent

To get a block from a library optimizer directly, give it every row of your data, then ask it
for points:

```python
# next_block.py: run from the repository root with .agent-venv/bin/python next_block.py data.csv
import csv, json, sys
sys.path.insert(0, "src/agentoptimizer/loop/workspace/lib")
from turbo import Optimizer  # or cma_es, de, sobol, random_search; see below

rows = list(csv.DictReader(open(sys.argv[1])))  # columns x0, x1, ..., y
dim = sum(c.startswith("x") for c in rows[0])
X = [[float(r[f"x{i}"]) for i in range(dim)] for r in rows]
y = [float(r["y"]) for r in rows]  # minimized: negate a value you want to maximize
space = {"dim": dim, "bounds": [[-5, 5]] * dim,  # your real bounds, one [low, high] per x
         "n_constraints": 0, "noisy": False, "indexed_pool": False, "candidates": None}

opt = Optimizer(space, seed=0)
opt.tell(X, y, [[] for _ in y])  # every row of your data
for x in opt.ask(5):             # the block: 5 points to evaluate
    print(json.dumps(x))
```

Each library optimizer's settings are in the docstring of its file in
[`lib/`](src/agentoptimizer/loop/workspace/lib/README.md); pass them as keyword arguments to
`Optimizer`.

The library is sequential: each optimizer expects the value of one point before it proposes
the next. Asked for several points at once:
- `turbo`, `cma_es`, `de`, `sobol` and `random_search` return distinct points.
- `gp_bo` and `nelder_mead` repeat the same point, so ask them for one point at a time.
- `pso` returns some duplicates.

For a fixed set of candidates, set `"indexed_pool": True` and `"candidates"` to the feature
rows. `ask` then returns row indices, and `tell` takes indices instead of points.

## Change the agent's strategy

The agent's behaviour comes from three places, in order of how much they change:

- **The skill**, [`loop/workspace/SKILL.md`](src/agentoptimizer/loop/workspace/SKILL.md). It
  is the agent's whole system prompt. It sets how to analyse the data, how to choose each
  block, the proposal format, and how to finish the budget. Edit it as plain text.
- **The library**, [`loop/workspace/lib/`](src/agentoptimizer/loop/workspace/lib/README.md),
  is the menu of optimizers the agent can name.
  - **Add an optimizer:** subclass `_base.Base` (or `_local.Local` for a solver that calls the
    objective itself). Then add a row to `lib/README.md`, since that table is how the agent
    finds it.
  - **Keep `lib/` self-contained:** files there use flat imports (`from _base import Base`)
    and never import this package. Add any new package to the `agent` group:
    `uv add --group agent <pkg>`.
  - **Agent-written optimizers:** the agent can also write its own in `work/` during a run.
    `SKILL.md` explains how.
- **Runner flags** on `python -m agentoptimizer.loop`: `--model`, `--effort`, `--agent` and
  `--context persistent`. With `persistent`, Claude keeps one session across calls instead of
  starting fresh each time. Rules on proposals are in `Campaign.validate` and
  `Campaign.prompt` in [`campaign.py`](src/agentoptimizer/loop/campaign.py): the accepted
  fields, the block-length limit, and three failed calls before the runner stops.

`SKILL.md` and `lib/` are part of the experimental condition: changing them changes what every
later run measures. Commit the change and say why in the commit message. Each run's
`private/manifest.json` records the commit and any uncommitted changes, so a run made from an
edited tree can be told apart from one made from a clean commit. Check a change with
`--check-only`, then a short run such as `--budget 40` on a cheap task.

## Baselines

One task, one optimizer, one seed:

```bash
.venv/bin/python -m agentoptimizer.baselines --task step_ellipsoid10 --optimizer cma_es --seed 0 --budget 100 --outdir runs/baselines-100
```

Optimizers: `bobyqa`, `cma_es`, `cobyla`, `de`, `gp_bo`, `lbfgsb`, `nelder_mead`,
`random_search`, `smac`, `sobol`, `turbo`, `pool_random` (pool tasks only). Indexed pools
(`mof_pool`) use the exact-row `gp_bo`, `turbo` and `pool_random`. Each campaign
writes `<outdir>/<task_name>/<optimizer>/seed_<seed>/{trace.csv,meta.json}`.

## Layout

```text
src/agentoptimizer/
  loop/              the runner: campaign, sandbox, preflight, agent CLIs, audit
    workspace/       copied as-is into every sandbox: SKILL.md, check.py, host.py, lib/
  baselines/         baseline optimizers and their campaign runner
  tasks/             objective implementations and the registry (private)
data/                benchmark inputs and the pipelines that produced them
external/, runs/     git-ignored private data and outputs
```
