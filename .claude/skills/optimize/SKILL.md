---
name: optimize
description: Run the optimization agent on a benchmark task in one call. Sets up the environments if needed, launches one isolated campaign per seed in the background (each passes the isolation preflight first), and reports the results when they end. Also lists tasks and reports on runs under runs/. Use when asked to run the agent, start a campaign or experiment on a task, or check on runs.
argument-hint: "<task> <budget> [seeds 0 1 2] [agent claude|codex] [model …] [effort …] [block size …] [round size …] [baselines] | status [run …] | tasks"
allowed-tools: Bash(.venv/bin/python scripts/agent_runs.py:*), Bash(.venv/bin/python scripts/run_sweep.py:*), Bash(uv sync:*), Bash(UV_PROJECT_ENVIRONMENT=.agent-venv uv sync:*), Bash(command -v:*), Bash(codex login status)
---

# Run the optimization agent

Arguments: $ARGUMENTS

Work from the repository root. Every command below goes through
`scripts/agent_runs.py`, which wraps `scripts/run_agent.py`; see the README
for what a run contains.

## 1. Read the request

- **No arguments, or `tasks`:** run `.venv/bin/python scripts/agent_runs.py tasks`,
  show the keys, and ask for a task and budget. Stop there.
- **`status`, optionally with run directories:** run
  `.venv/bin/python scripts/agent_runs.py status [run …]`, adding
  `--baselines runs/baselines` when run directories are given and that folder
  exists. Summarize the output. Stop there.
- **Anything else is a launch.** Read these settings from the arguments, in any
  order or wording:

| Setting | `launch` flag | Default |
| --- | --- | --- |
| Task: a key from `agent_runs.py tasks` | `--task` | required |
| Budget: objective evaluations, not tokens | `--budget` | required |
| Seeds, one independent campaign each | `--seeds 0 1 2` | `0` |
| Agent CLI | `--agent claude` or `--agent codex` | `claude` |
| Model | `--model` | `claude-opus-5-5` (claude), `gpt-6-astra` (codex) |
| Reasoning effort | `--effort low`…`max` | `medium` |
| Suggested block size, written to `problem.toml` | `--block-size` | 10% of the budget |
| Evaluations per round before results come back | `--round-size` | `1` |
| Baseline sweep for comparison | none (see step 3) | off |

A launch spends model budget, so ask instead of guessing when the task or the
budget is missing. If the task is not a registry key, show the closest keys
and ask. For reference, the paper's settings are the 30-task suite at 2000
evaluations and the OPV and MOF pools at 100 evaluations with seeds 0 1 2.
Constrained tasks are not supported yet; say so if one is asked for.

## 2. Check the setup (no model calls)

- If `.venv/bin/python` or `.agent-venv/bin/python` is missing, create them:

  ```bash
  uv sync --all-extras --locked
  UV_PROJECT_ENVIRONMENT=.agent-venv uv sync --only-group agent --locked
  ```

- `command -v codex` must succeed for every launch: the Codex CLI is the
  per-command sandbox for both agents. With `--agent claude`, `command -v claude`
  must succeed too. With `--agent codex`, `codex login status`
  must report that it is logged in; otherwise the run passes its preflight and
  then fails with 401 errors and zero evaluations. If a CLI is missing or
  signed out, tell the user to install it or run `codex login` themselves, and
  stop. Never sign in for them.

## 3. Launch

One call starts every seed:

```bash
.venv/bin/python scripts/agent_runs.py launch --task <task> --budget <budget> --seeds <seeds> [--agent …] [--model …] [--effort …] [--block-size …] [--round-size …]
```

It returns once each run has passed its isolation preflight (zero evaluations,
no model call yet) and prints each run directory. The runs are detached: they
keep going if this session ends. If a preflight fails, show the printed log
lines and stop; do not relaunch with a changed setup unless the user asks.

If the user asked for baselines, also start the sweep with Bash
`run_in_background` (no model calls; GP-BO and TuRBO are slow at large budgets):

```bash
.venv/bin/python scripts/run_sweep.py --task <task> --budget <budget> --seeds <highest seed + 1> --outdir runs/baselines --workers 4
```

## 4. Watch and report

Start the waiter with Bash `run_in_background`, so you are told when the runs end:

```bash
.venv/bin/python scripts/agent_runs.py wait <run directories> [--baselines runs/baselines]
```

Then tell the user briefly: the run directories and settings, that the runs
continue if this session closes, and that `/optimize status` reports progress
at any time. End the turn; do not poll.

When the waiter ends, report from its output, per seed: the state, evaluations
used, best value and gap to the optimum, the sequence of method blocks, the
advisor's cost, the integrity audit and, with baselines, how many baselines the agent beat. If the
baseline sweep is still running, say so and suggest `/optimize status <run>`
later. For a run that did not finish, name its logs.

## Rules

- Do not read or change anything under a run's `workspace/` while it runs, and
  never pass information back to the agent. `agent_runs.py` reads only the
  private journal, read-only.
- Never delete or reuse a run directory. There is no resume; a stopped run is
  kept as evidence.
- Do not start `run_agent.py` some other way to work around a failed preflight.
