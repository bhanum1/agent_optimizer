# Axis study v5: confirmation on fresh instances, with TuRBO

Status: **2026-11-10, run unattended overnight on Trillium.** Predictions are frozen in
`src/aobench/tasks/AXIS_V5_PREDICTIONS.md`. Decisions made without anyone to ask are
marked **Decision**.

## Goal

The same three conditions as v4:
- every baseline fails (scores near Sobol) on at least one task;
- pairs of baselines swap order between tasks, with statistical support;
- every task has precedent in the optimization literature.

v4 found a failure for every solver it ran. The CMA-ES failures (f23 Katsuura,
f16 Weierstrass at d ≥ 20) were found after the fact. v5:
- re-tests every v4 failure on BBOB instances that v4 did not use (16–30
  instead of 1–15);
- adds TuRBO, with its own predicted failure (ill-conditioned functions) and
  predicted rank reversals against L-BFGS-B and CMA-ES.

## Solvers

`sobol` (the reference), `lbfgsb`, `nelder_mead`, `cobyla`, `cma_es`, `turbo`.

- The first five run with `restart=True`, as in v4. `lbfgsb`, `nelder_mead`,
  `cobyla` and `cma_es` restart from uniform random points. Sobol has no restarts.
- **TuRBO and `restart`:** `campaign.run_campaign` only wraps solvers in
  `RESTARTABLE` = {lbfgsb, nelder_mead, cobyla, cma_es} in `Restarting`. TuRBO is not
  in that set, so `restart=True` leaves it untouched. It runs as the plain
  `baselines/turbo.py` TuRBO-1, with these settings:
  - q = 1, 2·d initial points;
  - a refit every 50 evaluations;
  - success/failure tolerance 3 / max(4, d);
  - its own restart, with fresh local data, when the trust region shrinks below
    0.5⁷.

  Its meta.json therefore has no `n_restarts`.

## Task groups

Native COCO BBOB functions from `ioh` (code: `src/aobench/tasks/axis_v5.py`, keys
`v5_<group>_<fn>_d<dim>_b<beta>`). The noise is v4's seedable bbob-noisy Gaussian
model on the gap f − f*. The noise stream is seeded by
(seed, d, function, 2026, 5): v4 used 4 in the last place.

| Group | Functions | d | β | Hypothesis (who fails) | Basis |
|---|---|---|---|---|---|
| plateau | f7 step ellipsoid | 5, 20 | 0 | Nelder–Mead | v4: advantage −0.76, last on every seed |
| noise_moderate | f1, f9, f10 | 5, 20 | 0.01 | L-BFGS-B | v4: collapsed on 100% of cells; Moré & Wild 2009 |
| noise_severe | f1, f9, f10 | 5, 20 | 1 | COBYLA | v4: collapsed on 89% of cells; bbob-noisy, Hansen et al. 2009 |
| katsuura | f23 Katsuura | 5, 20 | 0 | CMA-ES (Nelder–Mead and COBYLA win) | v4, post hoc |
| weierstrass | f16 Weierstrass | 5, 20 | 0 | CMA-ES at d = 20 (COBYLA wins) | v4, post hoc |
| illcond | f10, f11, f12 | 5, 20 | 0 | TuRBO (L-BFGS-B wins) | Le Riche & Picheny 2021 (citation from memory) |

**Budget: fixed at 300·d.** The 10/30/100·d checkpoints are kept in the traces
for description only (the regret curves). No claim of the form "fails at a small
budget, succeeds at a large one" is made.

**Seeds:** 15–29 = BBOB instances 16–30.

## Cost and the d = 40 decision

TuRBO's GP is exact, fit on the current trust region's local data. Each step
also draws a Thompson sample over min(100·d, 5000) candidates. So cost grows
with the local data size n, which resets only when the trust region restarts.

Timing pilot (Trillium job 2825011, 24 TuRBO runs on one node, 1 thread each,
v4 task keys on instances 16–17; only wall time and n were read, never regret):

| d | Cells | Observed |
|---|---|---|
| 20 | noise, katsuura, weierstrass, plateau | trust region restarts every ~100–400 evals; **all 6,000 evals finished in 10–13 min** |
| 20 | illcond (f10, f12) | restarts every ~300–525 evals, n ≤ ~525; ≈ 0.5–1.2 s/eval, so **≈ 1–2 h per campaign** |
| 40 | illcond, plateau | **no restart by 800–900 evals**, n = evals; cost per 100 evals rose from 74 s (n ≈ 250) to 117 s (n ≈ 750) to 122 s (n ≈ 850) |
| 40 | noise, rugged | restarts every ~300–500 evals |

A second probe (job 2825033) timed one TuRBO step on synthetic local data:

| d | n | Step with refit | Step without refit |
|---|---|---|---|
| 20 | 500 | 22 s | 0.17 s |
| 20 | 1,000 | 156 s | 0.31 s |
| 40 | 500 | 55 s | 0.8 s |
| 40 | 1,000 | 453 s | 1.15 s |

Refit cost grows roughly as n³. The probe was cancelled once the decision was clear.

Extrapolating to a d = 40 illcond campaign (12,000 evals):
- a **lower bound**, linear in n, already gives ≈ 120·50 s + 0.09 s · Σ n ≈ 19.5 h;
- the cubic GP terms make the real figure larger.

Running all campaigns in parallel doesn't help, because each campaign is
serial. Submitted at 21:40, it could not finish by 06:00, and likely not
within the 24 h job limit either.

**Decision: d = 40 is dropped for every solver on every task, as the brief
requires. The budget is not cut.** `DIMS = [5, 20]` in `axis_v5.py`. Weierstrass's
hypothesis H5 ("CMA-ES fails at d = 20 and 40") is therefore tested at d = 20 only.

Resulting size: 12 functions × 2 d = **24 cells × 6 solvers × 15 seeds = 2,160
campaigns** (360 TuRBO).

## Jobs on Trillium

Account `def-moosavi5`, partition `compute`, whole 192-core nodes, threads = 1,
`PYTHONWARNINGS=ignore` (gpytorch's jitter warnings filled the pilot log).
Outputs go to `benchmark/runs/axis_v5/`.

| Job | What | Size | Limit |
|---|---|---|---|
| `sweep_cheap.sbatch` | sobol, lbfgsb, nelder_mead, cobyla, cma_es | 1,800 campaigns, 1 node (v4 took < 5 min) | 1 h |
| `sweep_turbo.sbatch`, array 0–2 | TuRBO, `--shard i/3` of the cost-sorted list (d = 20 illcond first) | 120 campaigns per node, all at once | 7 h 30 min |
| `analyze.sbatch` | `--dependency=afterany` on all of the above | runs `axis_v5_analyze`, `axis_v5_curves`, `axis_v5_plot` | 1 h |

**Decision:** the TuRBO limit is 7 h 30 min, so the analysis runs before 08:00
even if some campaign hangs. Campaigns cut off by the limit are listed as
incomplete in the report. If a sweep job dies early, it is resubmitted: finished
campaigns have a meta.json and are skipped. Each resubmission is logged below.

## Analysis

```
python -m aobench.baselines.axis_v5_analyze RUNS/axis_v5   # scores.csv, pairs.csv, report.md
python -m aobench.baselines.axis_v5_curves  RUNS/axis_v5   # curves.csv
python -m aobench.baselines.axis_v5_plot    RUNS/axis_v5   # plots/<group>.png
```

Definitions, verdict rules and the success criterion are in the predictions file.
Holm correction runs over the 15 solver pairs in each cell. Only the summaries
(`report.md`, `pairs.csv`, `curves.csv`, `plots/`) are copied back to
`benchmark/runs/axis_v5_summary/`. Raw traces stay on Trillium.

## Other decisions

- **Smoke tests:**
  - locally, 300 cheap-solver campaigns plus 15+ TuRBO campaigns at d = 5 (no errors);
  - the analyze, curves and plot scripts were run on the smoke output.

  That printed per-solver fail counts on the four d = 5 smoke cells in the report's
  section headers. This happened after the predictions file was written, and no
  prediction was changed.
- **Report "fails on a cell"** follows v4: advantage < 0.1 decades. "Who wins on that
  cell" is reported two ways: the solver with the best mean rank there, and the
  list of solvers that beat the failing one significantly.
- **Rank reversals in the predictions** go beyond the six hypotheses. They are R1–R8,
  including the two TuRBO pairs and four v4 replications.

## Run log

All times Toronto (EST).

- 21:24 pilot 2825011 and 21:28 probe 2825033 started; both cancelled at ~21:38 once
  the d = 40 decision was clear.
- 21:40 submitted: cheap sweep **2825083**, TuRBO array **2825084_[0-2]**, analysis
  **2825085** (`afterany:2825083:2825084`).
