# Benchmark

Two things run on the same black-box tasks:

- **The agent harness** (`python -m aobench.agent`) runs the skill with the harness
  playing the user.
- **Eight baseline optimizers** (`python -m aobench`), chosen so that each one has a
  task where it is good and a task where it is bad: no optimizer is safe everywhere.

Both write results in the same format, so the agent can be compared with the
baselines task by task.

## Tasks

BBOB functions from `ioh` on the box [-5, 5]^d. The seed picks the BBOB instance,
so each seed is a differently shifted and rotated copy. Noise, when present, is
COCO's multiplicative model on the gap to the optimum, `g · exp(β · N(0, 1))`.
Scores use the noise-free value of the best point observed.

| Task | Function | d | Budget | Good | Bad | Why |
|---|---|---|---|---|---|---|
| `ellipsoid` | f10, condition 1e6 | 5 | 1500 | L-BFGS-B | COBYLA, SMAC | smooth and unimodal, so gradients win |
| `noisy_ellipsoid` | f10, β = 0.01 | 5 | 1500 | CMA-ES | L-BFGS-B | 1% noise swamps finite-difference gradients |
| `very_noisy_sphere` | f1, β = 1 | 5 | 1500 | SMAC | L-BFGS-B, COBYLA | noise as large as the signal: local models fit the noise |
| `step_ellipsoid` | f7 | 5 | 1500 | CMA-ES | L-BFGS-B | plateaus, zero gradient almost everywhere |
| `noisy_sphere` | f1, β = 0.01 | 10 | 300 | COBYLA | L-BFGS-B, CMA-ES, SMAC | few evaluations: linear models converge through small noise; CMA-ES is too slow to start |
| `gallagher` | f21, 101 peaks | 2 | 60 | GP-BO | L-BFGS-B, COBYLA, CMA-ES | a global model finds the best peak; local solvers climb the nearest one |

The tasks are defined in [`src/aobench/tasks.py`](src/aobench/tasks.py). Add one by
adding an entry to `SPECS`, but only if it shows something the others do not.

## The agent harness

```bash
uv sync --locked
.venv/bin/python -m aobench.agent --out benchmark/runs/agent --tasks gallagher --seeds 1
.venv/bin/python -m aobench.agent --out benchmark/runs/try --tasks gallagher --agent fake
```

For each task and seed, the harness:

1. **Builds the workspace**: a fresh folder outside the repository with a copy of
   the skill, and the starting files `problem.yaml` and `PROBLEM.md`. They name no
   function: the variables are `x0`.., the objective is `y`, and the problem says
   its identity is withheld. Then it runs `aopt init`.
2. **Calls the agent**: a fresh, headless `claude -p`, told to follow the skill's
   METHOD.md. The agent writes a proposal, either an optimizer of its own or a list
   of points, and validates it.
3. **Runs the block** as a user would: `aopt commit` and `aopt suggest`, then it
   evaluates the task at every pending row of `data.csv`, writes the values in and
   runs `aopt suggest` again, until the block is done.
4. Repeats from step 2 until the budget is spent.

The agent and every `aopt` call (which runs the agent's own code) run in a
bubblewrap sandbox. It holds the workspace at `/work`, the skill's Python read-only
and an empty home, and nothing else: not the repository, the task definitions or
other results. So the `claude` agent needs Linux with `bwrap`. The `fake` agent is a
scripted stand-in that alternates the two kinds of proposal. It runs without a
sandbox, on any platform, and is how to test the harness without model calls.

Claude logs in with the `ANTHROPIC_*` settings in the repository's `.env`
(git-ignored), either an API key or a gateway URL and key. If there are none, it uses
this machine's claude.ai login.

Options: `--model`, `--effort`, `--workers` (campaigns at once), `--call-timeout`,
`--max-calls`, `--work` (where the workspaces go, default `~/.aobench/work`).

| Code | Role |
|---|---|
| [`agent/campaign.py`](src/aobench/agent/campaign.py) | one campaign: the workspace, the call-commit-measure loop, the result |
| [`agent/problem.py`](src/aobench/agent/problem.py) | the starting files for a task |
| [`agent/sandbox.py`](src/aobench/agent/sandbox.py) | the bubblewrap sandbox |
| [`agent/claude.py`](src/aobench/agent/claude.py) | the `claude -p` command, its login, and its log |
| [`agent/fake.py`](src/aobench/agent/fake.py) | the scripted stand-in agent |

Per campaign it writes `<out>/<task>/agent/seed_<s>.csv` and, in `seed_<s>/`, one
line per call in `calls.jsonl` (blocks, cost, time) and each call's prompt, event
stream and outputs in `calls/NNN/`.

## The baselines

| Key | Solver | Notes |
|---|---|---|
| `sobol` | scrambled Sobol sequence | the reference: "bad" means no better than this |
| `lbfgsb` | L-BFGS-B, finite-difference gradients (scipy) | restarts from a random point when it stops |
| `cobyla` | COBYLA, linear-model trust region (scipy/PRIMA) | restarts from a random point when it stops |
| `cma_es` | CMA-ES (`cma`) | plain restarts from random points, no IPOP/BIPOP |
| `smac` | SMAC3, random-forest BO (HPO facade) | stock settings |
| `gpbo` | GP-BO: one global GP with LogEI (BoTorch) | refits every evaluation, 2d Sobol start |
| `shiwa` | Nevergrad's Shiwa, a rule-based selector | told the budget and whether the task is noisy. It picks MetaCMA on deterministic tasks and TBPSA on noisy ones |
| `bandit` | UCB1 bandit over CMA-ES, Sobol, COBYLA, L-BFGS-B and GP-BO | details below |

The first six are single methods, each with its library's defaults. `shiwa` and
`bandit` choose among methods, as the agent does. They are reported, but the report
never counts them as the best solver on a task.

**The bandit** first gives each arm 25 evaluations (fewer when the budget is short).
Then UCB1 picks the arm for each further block of 25. A block's reward is the number
of decades by which the arm's best value moved toward a reference just below the
campaign's best: the best minus the spread of the best five. So the reward does not
depend on the objective's scale. Each arm's mean reward is taken relative to the best
arm's. Each arm is the baseline above, paused and resumed. It writes its picks to
`seed_<s>.json`.

The code is in [`src/aobench/baselines/`](src/aobench/baselines/), with the registry
in its `__init__.py`.

```bash
.venv/bin/python -m aobench --out benchmark/runs/main
```

- **Defaults:** every task and solver, seeds 0–9, all cores but one.
- **Narrow a run** with `--tasks`, `--solvers`, `--seeds N` and `--workers N`.
- **Re-print a finished run** with `--report-only`.

Finished campaigns are skipped, so an interrupted run picks up where it stopped.

Each campaign writes `<out>/<task>/<solver>/seed_<seed>.csv` with one row per
evaluation: `eval, y, regret`. Here `regret` is the noise-free regret of the best
point observed so far. The report goes to `<out>/report.md`. It gives the median
final log₁₀ regret per task and solver and marks the best solver. It also marks soft
failures (◦, at least one decade behind the best) and hard failures (✗, no better
than Sobol).
