# Preregistered predictions: benchmark task screen

**Status: FROZEN 2026-11-06,** by the commit that added the budget section
below, before any screening run on the candidate tasks. Later edits go in
the amendment log at the end and leave the original text in place.

The file lives in `tasks/`, so like everything else there it must never
reach an agent sandbox.

## Purpose

The benchmark is a small set of tasks, each built around one landscape
feature known to break at least one optimizer family. This file fixes,
before any data is seen:

1. the landscape axes, and where each candidate task sits on each one;
2. which baselines are predicted to succeed or fail on each task;
3. the rule for scoring success and failure;
4. the rule for choosing which candidates go into the benchmark.

Tasks are chosen with rule 4 from the screen's results. The predictions are
then scored against the same results; they are never used to choose tasks.

## Baselines

These names are the keys in `baselines/campaign.py`.

| Key | Family |
|---|---|
| `sobol` | Space-filling (the reference; no predictions) |
| `lbfgsb` | Gradient, quasi-Newton (finite-difference gradients) |
| `nelder_mead` | Local derivative-free, direct search |
| `bobyqa` | Local derivative-free, quadratic-model trust region |
| `cobyla` | Local derivative-free, linear models with explicit constraints |
| `cma_es` | Evolution strategy (plain, no restarts) |
| `de` | Differential evolution (NP = 10·d) |
| `gp_bo` | Global GP Bayesian optimization |
| `turbo` | Trust-region GP Bayesian optimization |
| `smac` | Random-forest Bayesian optimization |

Every baseline runs with its library defaults, as documented in its module.
The local solvers start from the centre of the box and are not restarted.
A run that stops early is scored on the evaluations it actually used.

## Axes

| Axis | Levels |
|---|---|
| **A1 Modality** | unimodal · multimodal, global structure · multimodal, weak or deceptive structure |
| **A2 Conditioning** | low (≤ 10) · moderate (10–10³) · high (≥ 10⁶) |
| **A3 Separability** | separable · non-separable |
| **A4 Smoothness** | smooth · non-smooth or rugged · piecewise constant (plateaus) |
| **A5 Noise** | none · severe |
| **A6 Dimension** | low (≤ 5) · medium (10–14) · high (30) |
| **A7 Variable type** | continuous · mixed-integer · categorical and conditional |
| **A8 Constraints** | none · inequality (count, number active at the optimum) |

## Candidate tasks and their axis positions

The registry key comes first, with the results folder name in parentheses.
A task's defining feature is in **bold**. Its other axes stay at their mildest
level, so that a failure can be put down to that one feature.

| # | Task | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 |
|---|---|---|---|---|---|---|---|---|---|
| T1 | `sep_rastrigin10` (BBOB f3) | multi, global | low | **separable** | smooth | none | 10 | cont. | none |
| T2 | `bent_cigar10` (BBOB f12) | unimodal | **high (10⁶)** | non-sep. | smooth | none | 10 | cont. | none |
| T3 | `rastrigin10` (BBOB f15) | **multi, global** | low | non-sep. | smooth | none | 10 | cont. | none |
| T4 | `schwefel10` (BBOB f20) | **multi, deceptive** | low | non-sep. | smooth | none | 10 | cont. | none |
| T5 | `gallagher21_10` (BBOB f22) | **multi, weak (21 peaks)** | moderate (≤10³) | non-sep. | smooth | none | 10 | cont. | none |
| T6 | `step_ellipsoid10` (BBOB f7) | unimodal | moderate (10²) | non-sep. | **piecewise constant** | none | 10 | cont. | none |
| T7 | `schaffer30` (CEC-2017 F6, box ±5) | multi, global | low | non-sep. | rugged | none | **30** | cont. | none |
| T8 | `noisy_rosenbrock_unif5` (bbob-noisy f111) | unimodal | moderate | non-sep. | smooth | **severe (uniform)** | 5 | cont. | none |
| T9 | `pymoo_g07` (G07) | unimodal (quadratic) | moderate | non-sep. | smooth | none | 10 | cont. | **8 (3 linear, 5 nonlinear), 6 active** |
| T10 | `constrained_rastrigin10` (bbob-constrained f44) | **multi, global** | low | non-sep. | smooth | none | 10 | cont. | **3 linear** |
| T11 | `mixint_rosenbrock10` (bbob-mixint f8) | unimodal | moderate | non-sep. | piecewise constant (integer coordinates) | none | 10 | **8 integer + 2 continuous** | none |
| T12 | `yahpo_xgboost` (YAHPO rbv2_xgboost, OpenML 40981) | unknown | unknown | non-sep. | piecewise constant (in its integer and categorical coordinates) | none (surrogate) | 14 | **2 categorical, 8 conditional** | none |

Two tasks deliberately combine features: T10 (multimodality together with
constraints), and T7 (high dimension together with ruggedness, since CEC F6
is the only 30-dimensional candidate).

Notes:
- T8: COCO's noise generator cannot be seeded. Every campaign sees the same
  noise sequence, indexed by evaluation number, so on T8 the variation
  between seeds comes from the optimizer alone. Scoring uses the noise-free
  value of the best *observed* point (`y_true`).
- T9, T10: scoring uses the best *feasible* value. A run with no feasible
  point counts as a failure.
- T11, T12: integer and categorical coordinates are rounded inside the task.
  Every baseline, SMAC included, sees one continuous box.

## Success and failure criteria

For each (task, baseline) pair, take the final best value of each seed and
compute the log-regret

    L = log10(max(y_best − f_opt, 1e-8))

Let `M(b)` be baseline b's median L over seeds. A deterministic baseline
runs once, and that run is its median. Let `M_sobol` be Sobol's median, and
`M_best` the lowest median among all ten baselines. The normalized score is

    N(b) = (M_sobol − M(b)) / (M_sobol − M_best)

N = 1 means b is the best baseline. N ≤ 0 means b did no better than Sobol.

- **Success (S):** N ≥ 0.75.
- **Failure (F):** N ≤ 0.25.
- **Intermediate:** anything in between. A prediction of S or F that lands
  here counts as a miss, but the report lists it separately from the
  opposite outcome (S predicted, F observed, or the reverse).

## Predictions

S = predicted success, F = predicted failure, · = no prediction. These
predictions assume the budget fixed below. Changing the budget before the
freeze means revisiting this table.

| Task | lbfgsb | nelder_mead | bobyqa | cobyla | cma_es | de | gp_bo | turbo | smac | Rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 sep_rastrigin10 | F | F | · | · | · | F | · | S | S | The local solvers converge to the local minimum nearest the centre. DE gets too few generations at this budget. Axis-aligned forest splits exploit separability. |
| T2 bent_cigar10 | S | · | S | · | S | F | · | · | F | Smooth and unimodal: local models and covariance adaptation handle the conditioning. Random-forest and DE moves do not follow the rotated narrow direction. |
| T3 rastrigin10 | F | F | F | · | S | · | · | S | · | Local solvers stop in the first basin. A large initial step size and trust-region restarts follow the global trend. |
| T4 schwefel10 | F | F | F | · | F | · | S | F | S | The best regions lie near the box boundary, away from the centre start and away from the first good region found. A global surrogate keeps sampling broadly. |
| T5 gallagher21_10 | F | F | F | · | · | · | S | S | · | Whichever peak the centre start falls in, it is unlikely to be the global one. Methods that sample globally find better peaks. |
| T6 step_ellipsoid10 | F | · | F | F | S | · | · | · | S | Finite-difference gradients and linear or quadratic models are zero on plateaus. The forest models piecewise-constant responses. |
| T7 schaffer30 | · | F | · | · | S | · | F | S | F | Global GP and forest surrogates degrade at 30 dimensions. Local trust regions and CMA-ES scale. Nelder-Mead degrades with dimension. |
| T8 noisy_rosenbrock_unif5 | F | F | F | F | S | · | S | · | · | Finite differences and interpolating models fit noise, and the best observed point is chosen on a noisy value. Population averaging and a GP noise model are robust to it. |
| T9 pymoo_g07 | · | · | · | S | · | F | F | · | F | Six active constraints: methods that see only the penalized merit spend the budget near the feasible boundary. COBYLA models the constraints directly. |
| T10 constrained_rastrigin10 | F | · | · | F | S | · | · | · | · | Constraint handling does not help a local solver in a multimodal landscape. |
| T11 mixint_rosenbrock10 | F | · | · | · | S | · | · | · | S | Finite-difference steps never change a rounded integer coordinate. |
| T12 yahpo_xgboost | F | · | · | F | · | · | · | · | S | Piecewise constant in its categorical coordinates and flat in inactive conditional ones, which forests are designed to model. |

Each non-reference baseline has at least one predicted failure:
lbfgsb (T1, T3–T6, T8, T10–T12), nelder_mead (T1, T3–T5, T7, T8),
bobyqa (T3–T6, T8), cobyla (T6, T8, T10, T12), cma_es (T4), de (T1, T2, T9),
gp_bo (T7, T9), turbo (T4), smac (T2, T7, T9).

## Benchmark selection rule

The screen runs every baseline on every candidate at the fixed budget, with
seeds 0–4 (one seed for deterministic baselines). A candidate enters the
benchmark if both of the following hold:

1. **It separates the field:** M_sobol − M_best ≥ 1. The best baseline beats
   Sobol by at least one decade of regret.
2. **Some family fails:** at least one baseline other than Sobol has N ≤ 0.25.

If more than one candidate passes for the same defining axis (T4 and T5 both
target weak global structure), keep the one with the larger
M_sobol − M_best. Keep both only if their sets of failing baselines differ.

Agent runs (Sonnet and Opus) are run only on the tasks selected.

## Prior exposure

These candidates were run by the agent before this file was written, so
predictions about them are not blind:

- `rastrigin10` (T3): runs/opus_high_2000
- `sep_rastrigin10` (T1): runs/opus_high_2000

Baseline runs on candidates before writing: none in this session. Smoke tests
ran on non-candidate tasks only (`pymoo_g04`, `constrained_ellipsoid10`,
`rosenbrock10`, `yahpo_svm`). Add any earlier baseline runs on candidates here.

## Budget

**Primary budget: 50·d evaluations** per campaign. This is the expensive
black-box regime: it matches the range used in BO and HPO papers and the
upper end of COCO's view for expensive functions. It also keeps agent runs
affordable, since agent calls grow with the budget.

| Dimension | Tasks | Budget |
|---|---|---|
| 5 | T8 | 250 |
| 10 | T1–T6, T9–T11 | 500 |
| 14 | T12 | 700 |
| 30 | T7 | 1,500 |

- The predictions, the S/F criteria and the selection rule are all
  evaluated at the primary budget.
- **Secondary checkpoints (baselines only):** 10·d and 20·d, read from the
  same traces. They are reported as descriptive results only, not used for
  selection or for scoring predictions. Agent runs are not truncated this
  way: the agent plans around the budget it is told.
- **Expected consequence:** DE (NP = 10·d) gets 5 generations and plain
  CMA-ES about 50. Long-run methods are therefore expected to look weak at
  this budget, and that is a property of the regime, not a defect of their
  implementations.
- **Seeds:** 0–4 for stochastic baselines; seed 0 only for deterministic
  ones (`deterministic = True`: lbfgsb, nelder_mead, bobyqa, cobyla).

## Amendment log

**A1 (2026-11-06): seeds and machine, made before any result was seen.**

- **Seeds:** stochastic baselines run seeds 0–2 instead of 0–4.
  Deterministic baselines are unchanged (seed 0 only). Everywhere this file
  says "seeds 0–4", read "seeds 0–2". Medians are over 3 runs.
- **Reason:** GP-BO, SMAC and TuRBO take about 1 s per evaluation, which put
  the full screen at 4–8 hours on the original machine.
- **Machine:** the screen moves from macOS (Apple M1 Pro, ARM) to Linux
  (Intel i7-13700K, x86-64), with the same locked package versions.
  Floating-point results are not expected to match the earlier machine bit
  for bit. Every baseline in the screen runs on the same machine.
- **Blindness:** a first launch on the original machine was stopped about
  20 minutes in, before any campaign had finished. No trace was ever
  written, so no result was seen. Its logs are kept, ignored by git, in
  runs/screen-50d-aborted-mac.
- **Unchanged:** the predictions, criteria, selection rule and budget.
