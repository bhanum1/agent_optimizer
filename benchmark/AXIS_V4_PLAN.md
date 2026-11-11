# Axis study v4: rank-reversal suite

Status: **2026-11-10, approved. Predictions are frozen in
`src/aobench/tasks/AXIS_V4_PREDICTIONS.md`.**

## Goal

A small suite where:
- every baseline fails (scores near Sobol) on at least one task;
- pairs of baselines swap order between tasks, with statistical support;
- every task has precedent in the optimization literature.

v3 varied axes in a full factorial. v4 instead picks task groups, each built
around one failure mechanism from the literature, so the predicted rank
reversals can be tested directly.

## Solvers

`sobol` (the reference), `lbfgsb`, `nelder_mead`, `cobyla`, `cma_es`. Restarts
and settings are as in v3.

Dropped, and why:

| Solver | Reason |
|---|---|
| `random_search` | Sobol and random were indistinguishable in v3 |
| `de` | Budget-limited at 300·d; its home task (separable multimodal) needs a low CR |
| `bobyqa` | Out of scope |
| `gp_bo`, `turbo`, `smac` | Cost |

Without the surrogate methods, the planned plateau, 10·d-budget and
high-precision tasks have no winner or no loser left, so they are not run.

## Task groups

The functions are COCO BBOB functions from `ioh`, natively rotated and
conditioned. Code: `src/aobench/tasks/axis_v4.py`.

| Group | Functions | d | Noise β | Mechanism | Precedent |
|---|---|---|---|---|---|
| illcond | f10 ellipsoid, f11 discus, f12 bent cigar | 5, 20, 40 | 0 | Curvature ratios of 10⁶ | Hansen & Ostermeier 2001; Hansen et al. 2009 (BBOB definitions) |
| noise | f1 sphere, f9 rotated Rosenbrock, f10 ellipsoid | 5, 20, 40 | 0.01, 1 | Finite-difference and model steps break | Hansen et al. 2009 (bbob-noisy); Moré & Wild 2009; Arnold & Beyer 2003 |
| highdim | f1, f9, f10 | 60, 100 | 0 | Simplex search stalls as d grows | Torczon 1989; Gao & Han 2012; Varelas et al. 2018 (bbob-largescale) |
| rugged | f16 Weierstrass, f18 Schaffer F7 (cond. 1e3), f23 Katsuura | 5, 20, 40 | 0 | Local optima everywhere | Hansen et al. 2010 (BBOB-2009 comparison) |

The rugged group uses functions that were not in v3, so it confirms v3's
post-hoc rugged finding instead of reusing it.

## Size and cost

There are 42 cells × 5 solvers × 15 seeds = 3,150 campaigns at 300·d. The
slowest campaign is COBYLA at d = 100, about 3.6 minutes on an M1 Pro. One
192-core Trillium node with a 1-hour limit is enough.

## Analysis

`python -m aobench.baselines.axis_v4_analyze RUNS/axis_v4` writes `scores.csv`,
`pairs.csv` and `report.md`. The tests and the success criterion are in the
predictions file.
