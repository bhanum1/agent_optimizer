# Agentoptimizer

An LLM chooses optimization methods, hyperparameters, and evaluation block lengths
for a black-box problem. It plans **one block at a time**, inspects its observations,
and decides again. The total evaluation budget is enforced by a separate evaluator.

[prompts/optimization_agent.md](prompts/optimization_agent.md) is the canonical agent
prompt used for the iterative Astra experiments; there is only one maintained prompt.
The automated isolated runner below drives either Codex CLI or Claude Code CLI
(`--agent codex|claude`); both use Codex's native sandbox for every agent command.

Results are not kept in this repository. Runs produced before it was created are
in the source repository at tag
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

The native runner has been checked on macOS with Codex CLI
`0.162.0-alpha.17.2` and `0.163.0`; `--agent claude` with Claude Code CLI `2.1.308`
(sandboxed by Codex `0.163.0`). It fails before inference if its isolation checks fail.
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
   .venv/bin/python scripts/run_agent.py --task step_ellipsoid10 --budget 100 --check-only
   ```

   It should end with `Isolation passed: … checks, 0/100 evaluations.`

## Run the agent

Start a fresh campaign:

```bash
.venv/bin/python scripts/run_agent.py \
  --task step_ellipsoid10 --budget 100 \
  --model gpt-6-astra --effort max
```

The defaults are `gpt-6-astra` and `max`, so both flags can be omitted.
Use `--effort xhigh` to retain the reasoning setting used in earlier experiments.

`--seed` is supplied to the agent for its stochastic designs and optimizers and
also seeds private observation noise where supported. Separate invocations with
`--seed 0`, `--seed 1`, and `--seed 2` create three independent campaigns.

`--budget` is the number of **objective evaluations**, not model tokens or optimizer
iterations. It is supplied in the public task input and enforced privately.
Initialization, repeated points, finite differences and line searches all count.
The agent chooses the next block's method, settings and length; continuations also
require a new decision. It can preserve a live optimizer process between blocks.

To run Claude instead, add `--agent claude` (default model `claude-opus-5-5`,
same `max` default effort; `--claude` overrides the CLI path). Claude Code runs headless
with no built-in tools, settings, memory, CLAUDE.md, skills or claude.ai connectors:
its only tools are the same sandboxed `run_command`/`write_stdin` MCP executor.

```bash
.venv/bin/python scripts/run_agent.py \
  --task step_ellipsoid10 --budget 100 --agent claude
```

For the other two recent experiments, use `--task sharp_ridge10` or
`--task weierstrass10`. Every invocation creates a new opaque `runs/run-<id>/`;
there is no resume/reset option. Runs may be launched from separate terminals in
parallel. `--runs-dir`, `--agent-python` and `--codex` override local paths.

The native runner accepts every task in [the task registry](src/agentoptimizer/tasks/registry.py).
The agent sees the same box as the baselines: discrete, categorical and conditional
structure stays hidden inside the objective. Constrained tasks return `(y, g)`,
feasible iff every `g <= 0`. Noisy tasks set `noisy` and are reseeded with `--seed`
(default 0); the agent sees only observations, and the private journal keeps the
noise-free values. See [TASKS.md](docs/TASKS.md) and [ADDING_TASKS.md](docs/ADDING_TASKS.md).

## What the agent can access

```text
Fresh Codex CLI context + canonical prompt + public task description
    -> native OS sandbox: optimizer libraries and this run's workspace
    -> public harness.evaluate(x): request file containing x
    -> separate trusted evaluator: objective, durable budget and journal
    -> numeric response returned to the agent
```

No VM is required. A fresh conversation or a different working directory alone
is insufficient: the runner applies explicit deny-by-default filesystem rules.
Agent commands can read their public workspace and optimizer runtime, and write
scripts in `work/`, temporary files, requests and decision logs. They cannot read
the repository, objective implementation, private evaluator state, previous runs,
or siblings. Network access, browser/app tools, memories, personal configuration,
and permission escalation are disabled for the campaign. The only execution tools
are the trusted `workspace.run_command` and `workspace.write_stdin` MCP tools,
which enforce the native sandbox on every command.

The CLI retains the credentials needed for model inference; those credentials
are not exposed to agent commands. The agent sees bounds, dimension, direction,
noise status, budget, and its own observations. It does not receive task identity,
optimum, baseline results or this README.

For an explicitly public candidate pool such as `--task mof_pool`, the agent also
gets an anonymous, feature-only `candidates.npy`. It selects exact unused rows
through `h.evaluate_index(index)`. Labels, material IDs, and global ranks remain
private. This differs from `opv_pool`, whose hidden nearest-untested mapping is
preserved. See [MOF task details](data/mof_pool/README.md).

Preflight attempts forbidden reads/writes, a symlink escape, network access and
process inspection on macOS. Checks are recorded before the first evaluation.
These are tested isolation controls, not a formal guarantee against OS or tool
vulnerabilities. Never run the optimization agent directly in the repository.

## Outputs and supervision

The launcher prints the run directory immediately. While it runs, inspect its
private CLI event log from your supervisor terminal; do not send benchmark hints
back to the agent.

```text
runs/run-<id>/
  workspace/
    AGENTS.md                       generated copy of the canonical prompt
    PUBLIC_EVALUATOR.md              public problem and actual budget
    src/agent_harness/harness.py     transport client, no objective implementation
    work/                           agent-written scripts
    results/agent_run/
      roadmap_initial.md            FIRST BLOCK ONLY; immutable after recording
      changelog.md                  every subsequent decision, including continuations
      trace.csv                     actual observations, written by evaluator
      meta.json                     public run state, written by evaluator
  private/
    manifest.json                   source/prompt hashes, model and package versions
    isolation_checks.json           preflight results
    evaluations.sqlite              authoritative evaluations and decision snapshots
    cli_events.jsonl                CLI tool calls and messages (Codex or Claude stream-json)
    agent_final.txt                 final agent response
    integrity_audit.json            journal/artifact consistency checks
    scoring.json                   task identity and optimum, for offline analysis
    scored_trace.csv               y_best_so_far in the baseline convention (noise-free, feasible)
```

The historical filename `roadmap_initial.md` remains for compatibility; it is not
a full campaign roadmap. The evaluator snapshots decisions before evaluations,
rejects edits to the initial record and requires an append-only changelog.
Reconstructing the client does not reset the evaluation counter. On a crash or an
unknown evaluation outcome, preserve the run directory and inspect the private
journal; do not retry that campaign as though it had zero evaluations.

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

Always pass `--outdir`: the defaults (`--budget 2000 --seeds 10 --outdir results_linux`)
resolve relative to the current directory. Each campaign writes
`<outdir>/<task_name>/<optimizer>/seed_<seed>/{trace.csv,meta.json}`. Reruns skip
completed campaigns; a sweep whose existing campaigns used a different budget
errors before running anything.

Stochastic baselines use ten seeds; deterministic methods use one. Current sweeps
use the requested budget for every method. Historical `results_linux/` runs
may have GP-BO, TuRBO and BOBYQA capped at 100; inspect their metadata. The newest
full-budget benchmark is `results_full/` in the source repository. Numerical results can
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
  --agent-run runs/run-<id>/workspace/results/agent_run \
  --outdir runs/run-<id>/figures
```

`--task` is the results folder name (`<task_name>`, listed per key in
[TASKS.md](docs/TASKS.md)), not the registry key. To compare against the
published baselines instead of your own sweep, copy `results_linux/` from the
source repository to the repository root and pass `--results results_linux`.
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
.agent-venv/bin/python scripts/figures/make_global_metric_plots.py --results results_linux
```

Figures that leave the repo follow [FIGURES.md](docs/FIGURES.md). Their scripts
(`make_crossover_plot.py`, `make_agent_switching_plot.py`,
`make_agent_switching_gif.py`, `make_landscape_plot.py`) write into
`results_linux/`; each docstring names its outputs and is run from the repository root.

## Task-specific guides

- [OPV device pool (`opv_pool`)](data/opv_pool/README.md): rebuild the pool,
  baselines, agent runs and plots.
- [MOF exact candidate pool (`mof_pool`)](data/mof_pool/README.md): data
  provenance, agent campaigns and pool-aware baselines.
- [Tasks](docs/TASKS.md): what each registered task tests and its baseline winners.
- [Adding a task](docs/ADDING_TASKS.md): registration, backends and the public contract.

## Repository layout

- `prompts/optimization_agent.md`: the canonical agent prompt, copied into each
  workspace as `AGENTS.md`.
- `src/agentoptimizer/tasks/`: private objective implementations and registry.
- `src/agentoptimizer/optimizers/`: shared optimizer wrappers.
- `src/agentoptimizer/agent_harness/harness.py`: public evaluation client.
- `src/agentoptimizer/blind_runner/`: native execution, private evaluator and orchestration.
- `scripts/`: agent and baseline runners; `scripts/figures/`: plotting entry points.
- `data/`: task inputs (`mof_pool/`, `opv_emulator/`, `opv_pool/`) and the
  pipelines that built them.
- `docs/`: task catalogue, adding tasks, figure rules, and the manuscript plan.
- `tests/`: integrity tests for the evaluator and pool tasks.
- `AGENTS.md`, `CLAUDE.md`: contributor guide (not the experiment prompt).
- `external/`: ignored third-party data (YAHPO surrogates, user-supplied evaluators).
- `runs/`: ignored generated workspaces, logs and new results.

The integrity tests use fake objectives to check budget exhaustion, duplicate
request handling, crash recovery, configuration immutability and prospective logs.
