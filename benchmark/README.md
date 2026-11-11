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

BBOB functions from `ioh` on the box [-5, 5]^d. The seed picks the BBOB instance,
so each seed is a differently shifted and rotated copy. Noise, when present, is
COCO's multiplicative model on the gap to the optimum, `g · exp(β · N(0, 1))`.
Scores use the noise-free value of the best point the solver observed.

| Task | Function | d | Budget | Good | Bad | Why |
|---|---|---|---|---|---|---|
| `ellipsoid` | f10, condition 1e6 | 5 | 1500 | L-BFGS-B | COBYLA, TuRBO, SMAC | smooth and unimodal, so gradients win |
| `noisy_ellipsoid` | f10, β = 0.01 | 5 | 1500 | CMA-ES | L-BFGS-B | 1% noise swamps finite-difference gradients |
| `very_noisy_sphere` | f1, β = 1 | 5 | 1500 | SMAC | L-BFGS-B, COBYLA | noise as large as the signal: local models fit noise |
| `step_ellipsoid` | f7 | 5 | 1500 | CMA-ES | L-BFGS-B | plateaus, zero gradient almost everywhere |
| `noisy_sphere` | f1, β = 0.01 | 10 | 300 | COBYLA | L-BFGS-B, CMA-ES, SMAC | few evaluations: linear models converge through small noise, CMA-ES is too slow to start |

Median final log₁₀ regret, 10 seeds (lower is better; **bold** = best; ◦ = at least
1 decade behind the best; ✗ = no better than Sobol):

| Task | Sobol | L-BFGS-B | COBYLA | CMA-ES | TuRBO | SMAC |
|---|---|---|---|---|---|---|
| `ellipsoid` | 3.87 ◦ | **−5.84** | 2.73 ◦ | −5.20 | 2.46 ◦ | 3.15 ◦ |
| `noisy_ellipsoid` | 3.87 ◦ | 5.25 ✗ | 3.05 ◦ | **−6.13** | 2.37 ◦ | 3.09 ◦ |
| `very_noisy_sphere` | 1.04 ◦ | 1.49 ✗ | 1.06 ✗ | 0.04 | 0.36 | **−0.62** |
| `step_ellipsoid` | 0.89 ◦ | 1.24 ✗ | 0.16 ◦ | **−4.42** | −0.23 ◦ | 0.06 ◦ |
| `noisy_sphere` | 1.49 ◦ | 2.01 ✗ | **−6.03** | 0.40 ◦ | −1.73 ◦ | 0.64 ◦ |

Still to find: a task where TuRBO is the clear best. The tasks are defined in
[`src/aobench/tasks.py`](src/aobench/tasks.py). Add one by adding an entry to `SPECS`.

## Run

From the repository root:

```bash
uv sync --locked
.venv/bin/python -m aobench --out benchmark/runs/main
```

- **Defaults:** every task and solver, seeds 0–9, all cores but one.
- **Narrow a run** with `--tasks`, `--solvers`, `--seeds N` and `--workers N`.
- **Re-print a finished run** with `--report-only`.

Finished campaigns are skipped, so an interrupted run picks up where it stopped.

Each campaign writes `<out>/<task>/<solver>/seed_<seed>.csv` with one row per
evaluation: `eval, y, regret`, where `regret` is the noise-free regret of the best
point observed so far. The report goes to `<out>/report.md`. It gives the median
final log₁₀ regret per task and solver, marks the best solver, and marks soft (◦)
and hard (✗) failures.
