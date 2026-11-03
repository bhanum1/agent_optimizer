# Agentoptimizer

An LLM designs the optimizer for a black-box problem, **one block at a time**.
Each block, a fresh agent reads the data so far (a summary and the raw table)
and the decisions made before. It decides the next block's method,
hyperparameters and size, and writes `propose.py`: a script, built from the
`agentoptimizer.batch` proposers or from scratch, that fits the data and
proposes points. The lab measures `round_size` points at a time, and the
script proposes again after each round until the block is full. Then the
agent decides again.

The same loop serves real campaigns (`/opt-run`, people run the experiments)
and blind benchmarks (`run_agent.py`, a private evaluator plays the lab). The
policy is [prompts/advisor.md](prompts/advisor.md). The benchmark runner drives
either Claude Code CLI or Codex CLI (`--agent claude|codex`); both use Codex's
native sandbox for every agent command.

Results are not kept in this repository. Runs produced before it was created,
by the earlier live-session agent (one long session driving an evaluation
client, prompt `prompts/optimization_agent.md` at that tag), are in the source
repository at tag
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
   .venv/bin/python scripts/run_agent.py --task sphere10 --budget 100 --check-only
   ```

   It should end with `Isolation passed: … checks, 0/100 evaluations.`

## Run in one call

In Claude Code, from the repository root:

```text
/optimize sphere10 100 seeds 0 1 2
/optimize mof_pool 100 effort high baselines
/optimize status
```

The [skill](.claude/skills/optimize/SKILL.md) creates missing environments,
launches one detached campaign per seed once its isolation preflight passes,
waits for the runs to end, and reports each run's best value, gap to the
optimum, method blocks, advisor cost and integrity audit (with `baselines`, it also runs the
baseline sweep and counts the baselines the agent beat). The same steps from a
terminal:

```bash
.venv/bin/python scripts/agent_runs.py launch --task sphere10 --budget 100 --seeds 0 1 2
.venv/bin/python scripts/agent_runs.py status
.venv/bin/python scripts/agent_runs.py wait runs/run-<id> [--baselines runs/baselines]
```

`launch` passes other flags (`--agent`, `--model`, `--effort`, `--block-size`,
`--round-size`) to `run_agent.py`.
Its runs keep going after the terminal or Claude Code session closes; their
console output is in `private/launcher.log`. `status` reads each run's private
journal read-only and never touches the workspace. A run launched from inside a
Claude Code session does not inherit that session's `CLAUDE_*` variables: the
runner drops them, so the agent CLI starts as it would from a terminal.

## Advisor mode (`/opt-run`)

For real experiments rather than benchmarks: the objective is yours (a lab,
a simulation queue), and each block is proposed from the results so far.
In Claude Code, from the repository root:

```text
/opt-run init my_campaign                        # interview -> problem.toml; optional import of old results
/opt-run my_campaign                             # decide the next block, or propose its next round
/opt-run record my_campaign results.csv          # results back (or fill them in data.csv)
/opt-run status my_campaign
```

Each call decides one block (method, settings, size) under the policy in
[prompts/advisor.md](prompts/advisor.md), records why in `decisions.md`, and
writes `blocks/NNN/propose.py`: the block's `SIZE` and a deterministic
`propose(campaign, q)` that returns the next q points, usually by calling a
proposer from `agentoptimizer.batch`. The lab measures `round_size` points
at a time (`problem.toml`, default 1), so a block runs in rounds:
`propose()` is called once per round on the data so far, with no new
decision, and each round lands in `data.csv` as pending experiments. To try
it without a lab, use the [reaction example](examples/reaction/README.md)
and its simulated lab.

The skill drives `scripts/opt_run.py`, which also works on its own:

```bash
.venv/bin/python scripts/opt_run.py init my_campaign          # then edit my_campaign/problem.toml
.venv/bin/python scripts/opt_run.py import my_campaign old.csv # optional: earlier results, as block 0
.venv/bin/python scripts/opt_run.py new-block my_campaign      # opens blocks/001/ with summary.json
# write "## Block 1" in decisions.md and blocks/001/propose.py, then:
.venv/bin/python scripts/opt_run.py run my_campaign            # proposes the next round, validates it
.venv/bin/python scripts/opt_run.py accept my_campaign         # adds it to data.csv as pending
.venv/bin/python scripts/opt_run.py record my_campaign results.csv
# repeat run, accept, record until the block is full; then new-block again
.venv/bin/python scripts/opt_run.py summarize my_campaign
```

A campaign is a folder: `problem.toml` (parameters, objective, budget,
context; an optional `[pool]` of candidates), `data.csv` (one row per
experiment), `decisions.md`, and `blocks/NNN/` with each block's summary,
script and latest round. `propose()` reads the data through
`agentoptimizer.advisor.Campaign`; see
[campaign.py](src/agentoptimizer/advisor/campaign.py).

## Run the agent

One blind campaign on a registered task:

```bash
.venv/bin/python scripts/run_agent.py --task sphere10 --budget 100
.venv/bin/python scripts/run_agent.py --task mof_pool --budget 100 --block-size 10 --round-size 1
```

The runner puts an anonymous campaign (`x0`…, minimize `y`) in a sandboxed
workspace and plays the lab:

1. Each block, a fresh headless call follows the `/opt-run` steps under
   [prompts/advisor_headless.md](prompts/advisor_headless.md). It reads the
   campaign, appends its decision to `decisions.md`, writes `propose.py` and
   accepts the first round.
2. The runner refuses changed results, rows added other than by `accept`, and
   rewritten decisions. It reruns `propose.py` on the data the block saw and
   requires the same round.
3. It evaluates the round through the private evaluator, then runs
   `propose()` in the sandbox for each later round of the block, with no
   model call, until the block has `SIZE` experiments.

Defaults: `--agent claude` (model `claude-opus-5-5`), `--effort medium`,
`--block-size` 10% of the budget (the suggestion written to `problem.toml`;
the agent chooses), `--round-size 1`. `--agent codex` uses `gpt-6-astra`. Claude Code
runs headless with no built-in tools, settings, memory, CLAUDE.md, skills or
claude.ai connectors: its only tools are the sandboxed `run_command` and
`write_stdin` MCP executor.

`--seed` is written into the campaign for the agent's own seeds and also
seeds private observation noise where supported. `--budget` is the number of
objective evaluations; it is stated in `problem.toml` and enforced privately.
Every invocation creates a new opaque `runs/run-<id>/`; there is no resume.
`--runs-dir`, `--agent-python`, `--codex` and `--claude` override local paths.

Tasks: every unconstrained task in [the registry](src/agentoptimizer/tasks/registry.py).
Constrained tasks are not supported yet. Noisy tasks are reseeded with
`--seed`; the agent sees only observations, and the private journal keeps
the noise-free values. With an indexed pool such as `mof_pool`, the campaign
has a `[pool]` of anonymous, feature-only candidates (ids are row numbers),
each evaluable once. See [TASKS.md](docs/TASKS.md) and
[ADDING_TASKS.md](docs/ADDING_TASKS.md).

## What the agent can access

```text
Fresh CLI context + advisor policy and procedure
    -> native OS sandbox: optimizer libraries, a copy of agentoptimizer.advisor
       and agentoptimizer.batch, and this run's campaign folder
    -> the runner reads the accepted rows, reproduces them and evaluates them
       privately: objective, durable budget and journal never enter the sandbox
```

No VM is required. The runner applies explicit deny-by-default filesystem
rules: agent commands can read their workspace and optimizer runtime, and
write only `campaign/`, `work/` and temporary files. They cannot read the
repository, the objective implementation, private evaluator state, previous
runs or siblings. Network access, browser and app tools, memories, personal
configuration and permission escalation are disabled. The only execution
tools are the trusted `run_command` and `write_stdin` MCP tools, which
enforce the sandbox on every command. The agent never sees task identity,
optimum, baseline results or this README.

The preflight attempts forbidden reads and writes, a symlink and a hardlink
escape, network access and, on macOS, process inspection, before any model
call. These are tested isolation controls, not a formal guarantee against OS
or tool vulnerabilities.

## Outputs and supervision

```text
runs/run-<id>/
  workspace/
    AGENTS.md                 the advisor policy and headless procedure
    opt_run.py, lib/          the campaign CLI and its agentoptimizer copy
    campaign/                 problem.toml, data.csv, decisions.md, blocks/NNN/
  private/
    manifest.json             source hashes, model, effort, block and round size
    isolation_checks.json     preflight results
    campaign/                 the truth the workspace copy is checked against
    blocks/NNN/               each advisor call's prompt, events and logs; round logs
    blocks.csv                per block: size, rounds, turns, tokens, cost
    evaluations.sqlite        authoritative evaluations and decision snapshots
    integrity_audit.json      journal and data agree; complete, or why it stopped
    scoring.json              task identity and optimum, for offline analysis
    scored_trace.csv          y_best_so_far in the baseline convention (noise-free, feasible)
```

`decisions.md` is journaled with each block and must only grow. On a crash
or an unknown evaluation outcome, keep the run directory and inspect the
private journal; do not rerun the campaign as though it had no evaluations.

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

`scripts/run_batch_baselines.py` runs fixed batch methods at a block size q
through the same `agentoptimizer.batch` proposers the agent uses: Sobol then
qLogNEI, Sobol then a TuRBO-style trust region, and random.

On a SLURM cluster, [`scripts/slurm/sweep_all.sh`](scripts/slurm/sweep_all.sh)
runs `run_sweep.py` for several tasks on one node. Submit it from the repository root:

```bash
sbatch -p debug scripts/slurm/sweep_all.sh sphere10 rastrigin10
```

## Figures

`make_comparison_plots.py` reads a legacy live-session run's
`results/agent_run/` folder; it does not read loop runs yet. For a legacy run:

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

- `prompts/advisor.md`: the policy every advisor follows;
  `prompts/advisor_headless.md`: the per-block procedure of a headless advisor.
- `src/agentoptimizer/tasks/`: private objective implementations and registry.
- `src/agentoptimizer/optimizers/`: ask/tell wrappers the baselines run.
- `src/agentoptimizer/blind_runner/`: the blind runner: `loop.py` (blocks and
  rounds, checks, reproduction), `agent_cli.py` (headless advisor),
  `campaign.py` (the workspace campaign), `evaluator.py`, `execution.py` and
  `preflight.py` (sandbox), `runner.py` (`scripts/run_agent.py`).
- `scripts/`: agent and baseline runners, `agent_runs.py` to launch and report
  on agent runs; `scripts/figures/`: plotting entry points.
- `.claude/skills/`: the `/optimize` and `/opt-run` Claude Code skills.
- `examples/`: advisor-mode example campaigns with simulated labs.
- `src/agentoptimizer/advisor/`: advisor mode (campaign folders, summary,
  proposal blocks); `scripts/opt_run.py` is its command line.
- `src/agentoptimizer/batch/`: batch proposers that rebuild their state from
  the data on each call (space filling, qLogNEI, trust region, CMA-ES, DE).
- `data/`: task inputs (`mof_pool/`, `opv_emulator/`, `opv_pool/`) and the
  pipelines that built them.
- `docs/`: task catalogue, adding tasks, figure rules, and the manuscript plan.
- `tests/`: integrity tests for the evaluator and pool tasks.
- `AGENTS.md`, `CLAUDE.md`: contributor guide (not the experiment prompt).
- `external/`: ignored third-party data (YAHPO surrogates, user-supplied evaluators).
- `runs/`: ignored generated workspaces, logs and new results.

The integrity tests use fake objectives to check budget exhaustion, duplicate
request handling, crash recovery, configuration immutability and prospective logs.
