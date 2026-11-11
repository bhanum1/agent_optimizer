# Benchmark: every solver fails somewhere

A small benchmark with one point to make: no optimizer is safe everywhere. Six
standard baselines run on a handful of tasks, chosen so that each baseline has a
task where it is good and a task where it is bad.

## Solvers

| Key | Solver | Notes |
|---|---|---|
| `sobol` | scrambled Sobol sequence | the reference: "bad" means no better than this |
| `lbfgsb` | L-BFGS-B, finite-difference gradients (scipy) | restarts from a random point when it stops |
| `cobyla` | COBYLA, linear-model trust region (scipy/PRIMA) | restarts from a random point when it stops |
| `cma_es` | CMA-ES (`cma`) | plain restarts from random points, no IPOP/BIPOP |
| `turbo` | TuRBO-1, GP trust region (BoTorch) | restarts its own trust region |
| `smac` | SMAC3, random-forest BO (HPO facade) | stock settings |

Every solver uses its library's defaults. The code is in
[`src/aobench/baselines/`](src/aobench/baselines/).

## Tasks

BBOB functions from `ioh`, d = 5, box [-5, 5]^d, budget 300·d evaluations. The seed
picks the BBOB instance, so each seed is a differently shifted and rotated copy.
Noise, when present, is COCO's multiplicative model on the gap to the optimum,
`g · exp(β · N(0, 1))`. Scores use the noise-free value of the best point the
solver observed.

| Task | Function | Good | Bad | Why |
|---|---|---|---|---|
| `ellipsoid` | f10, condition 1e6 | L-BFGS-B | Sobol | smooth and unimodal, so gradients win |
| `noisy_ellipsoid` | f10, β = 0.01 | COBYLA, CMA-ES | L-BFGS-B | 1% noise swamps finite-difference gradients |
| `very_noisy_sphere` | f1, β = 1 | CMA-ES | COBYLA | noise as large as the signal: COBYLA's linear models fit noise |
| `step_ellipsoid` | f7 | CMA-ES | L-BFGS-B | plateaus, zero gradient almost everywhere |
| `katsuura` | f23 | COBYLA | CMA-ES, TuRBO | rugged at every scale, no trend to follow |

Still to find: a clear "good" task for TuRBO, and both tasks for SMAC. The tasks are
defined in [`src/aobench/tasks.py`](src/aobench/tasks.py). Add one by adding an
entry to `SPECS`.

## Run

From the repository root:

```bash
uv sync --locked
.venv/bin/python -m aobench --out benchmark/runs/main
```

- **Defaults:** every task and solver, seeds 0–9, d = 5, all cores but one.
- **Narrow a run** with `--tasks`, `--solvers`, `--seeds N` and `--dim D`.
- **Re-print a finished run** with `--report-only`.

Finished campaigns are skipped, so an interrupted run picks up where it stopped.

Each campaign writes `<out>/d<dim>/<task>/<solver>/seed_<seed>.csv` with one row
per evaluation: `eval, y, regret`, where `regret` is the noise-free regret of the
best point observed so far. The report goes to `<out>/d<dim>/report.md`. It gives
the median final log₁₀ regret per task and solver, marks the best solver, and
marks a solver ✗ (bad) when it is less than 0.1 decades ahead of Sobol (median
over seeds).
