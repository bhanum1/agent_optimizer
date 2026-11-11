# Axis study v4: predictions

**Status: FROZEN 2026-11-10,** before the sweep ran. One thing was seen first: a
one-seed smoke test of the code on three cells (`highdim_ellipsoid_d100`,
`noise_rosenbrockrot_d20_b1`, `rugged_katsuura_d5`). Its final regrets were
printed. Where it changed a prediction, the prediction says so. Later edits go
in the amendment log at the end, and the original text stays in place.

This file is in `tasks/`, so it must never reach an agent sandbox.

## Setup

- Design: [`axis_v4.py`](axis_v4.py) and `benchmark/AXIS_V4_PLAN.md`.
- Solvers: `sobol` (the reference), `lbfgsb`, `nelder_mead`, `cobyla`, `cma_es`.
  The local solvers and CMA-ES restart from uniform random points, as in v3.
- Budget: 300·d. 15 seeds, which are COCO instances 1–15.
- Analysis groups (5): `illcond` (9 cells), `noise_moderate` (β = 0.01, 9 cells),
  `noise_severe` (β = 1, 9 cells), `highdim` (6 cells), `rugged` (9 cells).

## Definitions

- **Score**: log₁₀ regret at 300·d, floored at 1e-8. Regret is the noise-free f
  at the best-observed point, minus f_opt.
- **A beats B in a cell**: a paired (by seed) two-sided Wilcoxon signed-rank
  test, Holm-corrected over the 10 solver pairs in that cell, gives p < 0.05,
  and the median of score(B) − score(A) is > 0.
- **A beats B on a group**: A beats B in at least ⅔ of the group's cells, and
  B beats A in none of them.
- **Rank reversal (A, B)**: A beats B on one group and B beats A on another.
- **Advantage**: the median over seeds of score(Sobol) − score(solver).
  A solver **collapses on a group** if its advantage is < 0.1 decades in at
  least 80% of the group's cells.

## Predicted rank reversals

| | Pair | A beats B on | B beats A on | Sources |
|---|---|---|---|---|
| R1 | L-BFGS-B vs CMA-ES | illcond | noise_moderate, noise_severe, rugged | Quasi-Newton on ill-conditioned quadratics; noise destroys finite-difference gradients (Moré & Wild 2009); v3 P3 |
| R2 | L-BFGS-B vs COBYLA | illcond | noise_moderate | COBYLA's initial trust radius averages over moderate noise (v3) |
| R3 | L-BFGS-B vs Sobol | illcond | noise_moderate | v3: L-BFGS-B was worse than random under moderate noise |
| R4 | Nelder–Mead vs Sobol | illcond | noise_moderate | v3: Nelder–Mead was worse than random under moderate noise |

## Predicted collapses

| Solver | Collapses on | Does not collapse on |
|---|---|---|
| L-BFGS-B | noise_moderate, noise_severe | illcond, highdim |
| Nelder–Mead | noise_moderate, noise_severe, rugged | illcond |
| COBYLA | noise_severe | illcond, noise_moderate |
| CMA-ES | **none** | every group |

- **CMA-ES has no predicted collapse.** The tasks aimed at it (plateaus,
  10·d budgets) were dropped along with their winners (SMAC, the BO methods).
  So this suite is not expected to meet the "every solver fails once" goal for
  CMA-ES. Whether it collapses on rugged is open: Katsuura is hard for every
  solver.
- **High dimension for Nelder–Mead.** The literature predicts a collapse
  (Torczon 1989; Gao & Han 2012). In the smoke test, Nelder–Mead was 1.2 decades
  ahead of Sobol on one d = 100 cell. So the prediction is weakened to: on
  highdim, Nelder–Mead is beaten by L-BFGS-B and CMA-ES, and a collapse is not
  predicted.
- **Rugged for L-BFGS-B and COBYLA.** v3 saw 78% and 58% of tasks collapsed,
  respectively. The prediction: CMA-ES beats both on rugged. No collapse is
  predicted.

## Success criterion for the suite

At least 3 of R1–R4 are confirmed, and every non-reference solver except CMA-ES
collapses on at least one group.

## Amendment log

### Amendment 1 (2026-11-10, before any v4 result was seen)

Job 2824824 was submitted but had not finished, and no output from it had been
read.

**New group `plateau`:** f7 step ellipsoid at d = 5, 20, 40, noiseless, with
the same 5 solvers, 15 seeds and 300·d budget. It runs separately, on a
workstation.

Predictions:
- **CMA-ES collapses on plateau.** Its step size shrinks below the plateau
  width and the search stalls (Hansen 2011, INRIA RR-7751; Hamano et al. 2022,
  CMA-ES with margin).
- **L-BFGS-B collapses on plateau.** Its finite-difference gradients are zero
  on the flat steps.
- No winner is predicted. This group exists so that CMA-ES has a predicted
  failure; it is not meant as a rank reversal.

**Budget is fixed at 300·d, and it is not an axis.** The traces keep their 10,
30 and 100·d checkpoints for description only. No prediction or claim of the
form "solver X fails at a small budget and succeeds at a large one" is made
or will be tested.

The success criterion gains: "CMA-ES collapses on plateau".
