# Agentoptimizer

An LLM chooses optimization methods, hyperparameters, and evaluation block lengths
for a black-box problem, one block at a time. A Python loop drives it:

1. Call the agent once, fresh, with [SKILL.md](src/agentoptimizer/loop/SKILL.md) as its prompt.
   It analyzes the data so far and writes a proposal: an optimizer (from the
   [library](src/agentoptimizer/loop/lib/README.md) or its own), its settings, and how many
   points to evaluate.
2. Validate the proposal and append the agent's decision to `decisions.md`.
3. Run that optimizer in a sandbox for exactly that many points. The runner
   evaluates every point itself and appends it to `data.csv`.
4. Repeat until the budget is spent.

The agent never touches the objective. Optimizer state is saved to disk between
blocks. The runner drives either Codex CLI or Claude Code CLI (`--agent codex|claude`);
both use Codex's native sandbox for every agent command and optimizer.

Results are not kept in this repository. Runs produced before it was created,
by the earlier live-session agent, are in the source repository at tag
[`pre-manuscript-snapshot`](https://github.com/bhanum1/Cheapest_opt_lunch/tree/pre-manuscript-snapshot):

- [Astra max results: 30 mathematical tasks, 2,000-evaluation allowance](https://github.com/bhanum1/Cheapest_opt_lunch/tree/pre-manuscript-snapshot/main/agent_runs/astra_max_2000_20261022)
- [OPV and MOF pool results: three seeds each, 100 evaluations, supplied baseline comparisons](https://github.com/bhanum1/Cheapest_opt_lunch/tree/pre-manuscript-snapshot/main/agent_runs/materials_astra_max_100_3seeds_20261022)
- [Updated OPV comparison with 10 random-start runs per local optimizer](https://github.com/bhanum1/Cheapest_opt_lunch/tree/pre-manuscript-snapshot/main/agent_runs/opv_random_starts_100_20261022)
- [Baseline sweeps (`results_linux/`, `results_full/`)](https://github.com/bhanum1/Cheapest_opt_lunch/tree/pre-manuscript-snapshot/main)

## Requirements

- macOS (checked) or Linux (has an execution path; validate it on the target machine).
- [uv](https://docs.astral.sh/uv/) 0.12 or later. It installs Python 3.11 if needed.
- [Codex CLI](https://github.com/openai/codex), for **both** agents: it is the
  per-command sandbox. Authenticate it (`codex login`) only for `--agent codex`.
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code), only for
  `--agent claude`; sign in with `claude auth login`.

The loop has been checked on macOS with Codex CLI `0.163.0` as the sandbox, with
`--agent claude` and `--agent codex`. It fails before inference if its isolation checks fail.
Other CLI versions must support the named permission profiles used by the runner.

## Install once

Run every command in this README from the repository root unless it says otherwise.
Keep two Python environments, both built from `uv.lock`: the agent gets optimizer
libraries only; only the private evaluator gets this package and objective
backends such as `ioh`. The runner refuses an agent runtime that has either.

```bash
uv sync --all-extras --locked
UV_PROJECT_ENVIRONMENT=.agent-venv uv sync --only-group agent --locked
codex login
```

The first command builds `.venv`: this package (editable), the baselines, every
objective backend and matplotlib. The second builds `.agent-venv` from the
`agent` dependency group alone, without the package. The lock pins the library
versions used for the published runs; see `[tool.uv]` in `pyproject.toml`.

Run scripts with `.venv/bin/python` as below rather than `uv run`. A bare
`uv sync` (no `--all-extras`) removes the objective backends from `.venv`.

### Task backends

The `backends` extra holds the packages for the non-BBOB task families. They
live only in `.venv`, because agent sandboxes must not see them:

| Tasks | Needs |
| --- | --- |
| `cec_rastrigin30`, `noncont_rastrigin30`, `schaffer30`, `schaffer50` | `opfunu==1.0.1`, `setuptools<81` |
| `pymoo_g04`, `pymoo_g07` | `pymoo==0.6.1.6` |
| `constrained_*`, `noisy_rosenbrock_*`, `mixint_*` | `coco-experiment==2.8.2` |
| `ng_noisy_*` | `nevergrad==1.0.12` |
| `yahpo_svm`, `yahpo_xgboost` | `onnxruntime==1.31.0` plus the surrogate files below |
| `psa_v10` | A user-supplied evaluator at `external/psa_v10/psa_v10_blackbox.py`; not distributed |

The YAHPO surrogates are git-ignored; download them into `external/yahpo_data/`:

```bash
for s in rbv2_svm rbv2_xgboost; do for f in config_space.json encoding.json model.onnx; do curl -sfL --create-dirs -o external/yahpo_data/$s/$f https://raw.githubusercontent.com/slds-lmu/yahpo_data/v1.0.2/$s/$f; done; done
```

Check that every public task now builds (takes a few seconds):

```bash
.venv/bin/python -W ignore -c "from agentoptimizer.tasks.registry import TASKS; [TASKS[k]() for k in sorted(TASKS) if k != 'psa_v10']; print(len(TASKS) - 1, 'tasks load')"
```

## Quick start

These three checks spend **no model calls**. Run them after installing:

1. Unit tests (fake objectives; budget, crash-recovery and logging integrity):

   ```bash
   .venv/bin/python -m unittest discover -s tests -v
   ```

2. A 20-evaluation baseline campaign, no agent involved:

   ```bash
   .venv/bin/python scripts/run_campaign.py --task sphere10 --optimizer random_search --seed 0 --budget 20 --outdir runs/smoke
   ```

3. Agent setup and isolation preflight, **zero objective evaluations**:

   ```bash
   .venv/bin/python scripts/run_loop.py --task step_ellipsoid10 --budget 100 --check-only
   ```

   It should end with `Isolation passed: … checks.`

The library tests (`test_loop_lib.py`) take about three minutes.

## Run the agent

```bash
.venv/bin/python scripts/run_loop.py --task step_ellipsoid10 --budget 100
```

The defaults are `--agent claude --model claude-opus-5-5 --effort high`. For Codex,
pass `--agent codex` (default model `gpt-6-astra`). `--call-timeout` limits each
agent call (default 3600 s).

`--seed` is given to the agent for its optimizers' randomness, and also seeds
private observation noise where supported. Separate invocations with `--seed 0`,
`--seed 1` and `--seed 2` create three independent campaigns.

`--budget` is the number of **objective evaluations**. Initialization, repeated
points, finite differences and line searches all count. Every invocation creates a
new opaque `runs/run-<id>/`; there is no resume option. Runs may be launched in
parallel. `--runs-dir`, `--agent-python`, `--codex` and `--claude` override local paths.

The runner accepts every task in [the task registry](src/agentoptimizer/tasks/registry.py).
The agent sees the same box as the baselines. Constrained tasks add `g0..` columns,
feasible iff every `g <= 0`. Noisy tasks are reseeded with `--seed`; the agent sees
only observations, and the private journal keeps the noise-free values. See
[TASKS.md](docs/TASKS.md) and [ADDING_TASKS.md](docs/ADDING_TASKS.md).

## What the agent can access

Each call is a fresh CLI with no built-in tools, settings, memory, skills or
connectors. Its one tool, `run_command`, runs a command under Codex's native
sandbox, which denies everything by default:

- **Readable:** the run's workspace (problem statement, `space.json`, `data.csv`,
  `decisions.md`, the library, earlier block scripts and state) and the
  optimizer-only Python runtime.
- **Writable:** `work/` (scripts and custom optimizers), `agent/` (its proposal,
  decision and notes) and a temporary directory.
- **Denied:** the repository, the objective, `private/`, other runs, the system
  temporary directory, and the network.

Optimizers run under a second policy that may write only `state/`. The agent's
runtime (`.agent-venv`) must not contain objective backends; the runner refuses
one that does. The agent does not receive task identity, the optimum, baseline
results or this README.

For an indexed pool such as `--task mof_pool`, `space.json` holds the anonymous,
feature-only candidates and optimizers propose row indices; labels, IDs and ranks
stay private. `opv_pool` keeps its hidden nearest-untested mapping and its
candidates private. See [MOF task details](data/mof_pool/README.md).

Preflight checks both policies before any model call: forbidden reads and writes,
a symlink escape, network access, and missing objective backends. These are tested
isolation controls, not a formal guarantee against OS or tool vulnerabilities.

## Outputs

```text
runs/run-<id>/
  results/
    trace.csv           every evaluation, in the baseline format (noise-free y_best_so_far)
    meta.json           task, budget, outcome, agent, audit
    decisions.md        every decision and runner note
  workspace/            what the agent saw: data.csv, decisions.md, blocks/, state/, work/, agent/
  private/
    journal.csv         evaluations including noise-free values and timing
    blocks.json         each accepted proposal and how its block ended
    calls/NNN/          each agent call's prompt, event log, proposal, decision, context.md
    blocks/NNN.log      each optimizer's stderr
    audit.json          the post-run checks
    manifest.json, policy.json, isolation_checks.json
```

After the loop, an audit checks that the run stayed within budget, `data.csv`
matches the private journal, `decisions.md` is exactly what the runner wrote, each
block follows its decision and stays within its length, and only the sandboxed
tool was used. A failed audit stops before `results/` is written.

## Baselines

One task, one optimizer, one seed:

```bash
.venv/bin/python scripts/run_campaign.py \
  --task step_ellipsoid10 --optimizer cma_es --seed 0 --budget 100 \
  --outdir runs/baselines-100
```

Optimizers: `bobyqa`, `cma_es`, `de`, `gp_bo`, `lbfgsb`, `nelder_mead`,
`random_search`, `sobol`, `turbo`, and `pool_random` (pool tasks only). Each uses
its own standard defaults; `run_campaign.py` exposes no hyperparameters by design.

Run all standalone optimizers at the same budget into a **new** results directory:

```bash
.venv/bin/python scripts/run_sweep.py \
  --task step_ellipsoid10 --budget 100 --seeds 10 --workers 4 \
  --outdir runs/baselines-100
```

Always pass `--outdir`: the defaults (`--budget 2000 --seeds 10 --outdir results_full`)
resolve relative to the current directory. Each campaign writes
`<outdir>/<task_name>/<optimizer>/seed_<seed>/{trace.csv,meta.json}`. Reruns skip
completed campaigns; a sweep whose existing campaigns used a different budget
errors before running anything.

Stochastic baselines use ten seeds; deterministic methods use one. Current sweeps
use the requested budget for every method. Historical `results_full/` runs (in the
source repository) may have GP-BO, TuRBO and BOBYQA capped at 100; inspect their
metadata. Numerical results can
vary by platform and library version; use matched environments for formal comparisons.

On a SLURM cluster, [`scripts/slurm/sweep_all.sh`](scripts/slurm/sweep_all.sh)
runs `run_sweep.py` for several tasks on one node. Submit it from the repository root:

```bash
sbatch -p debug scripts/slurm/sweep_all.sh sphere10 rastrigin10
```

## Figures

After the agent finishes, compare it with the baselines. Replace `<id>` with the
printed run ID:

```bash
.agent-venv/bin/python scripts/figures/make_comparison_plots.py \
  --results runs/baselines-100 --task step_ellipsoid_d10_i1 \
  --agent-run runs/run-<id>/results \
  --outdir runs/run-<id>/figures
```

`--task` is the results folder name (`<task_name>`, listed per key in
[TASKS.md](docs/TASKS.md)), not the registry key. To compare against the
published baselines instead of your own sweep, copy `main/results_full/` from the
source repository to the repository root and pass `--results results_full`.
Figures go to `runs/run-<id>/figures/<task_name>/`.

This produces baseline convergence, a final-regret comparison, and an agent
method/decision timeline. The supplied agent's budget sets the comparison horizon;
longer baseline traces are truncated and early-stopped incumbents are carried
forward within their allowed budget. Historical prefix comparisons are not fresh
baseline reruns configured for the smaller budget. Scoring happens outside the
agent's workspace, using the private optimum from baseline metadata.

The cross-task metric (regret relative to random search, averaged over tasks)
reads a results folder under the repository root and writes `global_metric_vs_budget.png` into it:

```bash
.agent-venv/bin/python scripts/figures/make_global_metric_plots.py --results results_full
```

Figures that leave the repo follow [FIGURES.md](docs/FIGURES.md). Their scripts
(`make_crossover_plot.py`, `make_agent_switching_plot.py`,
`make_agent_switching_gif.py`, `make_landscape_plot.py`) write into
`results_full/`; each docstring names its outputs and is run from the repository root.

## Task-specific guides

- [OPV device pool (`opv_pool`)](data/opv_pool/README.md): rebuild the pool,
  baselines, agent runs and plots.
- [MOF exact candidate pool (`mof_pool`)](data/mof_pool/README.md): data
  provenance, agent campaigns and pool-aware baselines.
- [Lab fluid mixtures (`fluid_mixtures`)](data/fluid_mixtures/README.md): sdagent's
  measured mixtures, RDKit descriptors and pump work.
- [Viscosity excess pool (`viscosity_excess`)](data/viscosity_excess/README.md): the same
  mixtures' viscosity below ideal mixing.
- [Heat-capacity excess pool (`cp_excess`)](data/cp_excess/README.md): campaign 11's
  heat capacity above ideal mixing, outliers removed.
- [Tasks](docs/TASKS.md): what each registered task tests and its baseline winners.
- [Adding a task](docs/ADDING_TASKS.md): registration, backends and the public contract.

## Repository layout

- `src/agentoptimizer/loop/`: the agent loop: runner, sandbox, optimizer host, `SKILL.md`, audit.
- `src/agentoptimizer/loop/lib/`: the agent's optimizer library.
- `src/agentoptimizer/tasks/`: private objective implementations and registry.
- `src/agentoptimizer/optimizers/`: the baselines' optimizer wrappers.
- `scripts/`: agent runner (`run_loop.py`), baseline runners; `scripts/figures/`: plotting entry points.
- `data/`: task inputs and the pipelines that built them.
- `docs/`: task catalogue, adding tasks, figure rules, and the manuscript plan.
- `tests/`: tests for the loop, its library and the pool tasks.
- `AGENTS.md`, `CLAUDE.md`: contributor guide (not the experiment prompt).
- `external/`: ignored third-party and private data (YAHPO surrogates, lab data, user-supplied evaluators).
- `runs/`: ignored generated workspaces, logs and new results.
