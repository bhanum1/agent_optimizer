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
| `fluid_mixtures`, `viscosity_excess`, `cp_excess` | private CSVs in `external/fluid_mixtures/` |
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
.venv/bin/python scripts/run_loop.py --task step_ellipsoid10 --budget 100 --check-only
```

Then run:

```bash
.venv/bin/python scripts/run_loop.py --task step_ellipsoid10 --budget 100
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

## Baselines

One task, one optimizer, one seed:

```bash
.venv/bin/python scripts/run_campaign.py --task step_ellipsoid10 --optimizer cma_es --seed 0 --budget 100 --outdir runs/baselines-100
```

Optimizers: `bobyqa`, `cma_es`, `de`, `gp_bo`, `lbfgsb`, `nelder_mead`,
`random_search`, `sobol`, `turbo`, `pool_random` (pool tasks only). Indexed pools
(`mof_pool`) use the exact-row `gp_bo`, `turbo` and `pool_random`. Each campaign
writes `<outdir>/<task_name>/<optimizer>/seed_<seed>/{trace.csv,meta.json}`.

## Layout

- `src/agentoptimizer/loop/`: runner, sandbox, preflight, optimizer host, `SKILL.md`, audit.
- `src/agentoptimizer/loop/lib/`: the agent's optimizer library, copied into each sandbox.
- `src/agentoptimizer/optimizers/`: the baseline optimizers.
- `src/agentoptimizer/tasks/`: objective implementations and the registry (private).
- `scripts/`: `run_loop.py` (agent campaign), `run_campaign.py` (baseline campaign).
- `data/`: benchmark inputs and the pipelines that produced them.
- `external/`, `runs/`: git-ignored private data and outputs.
