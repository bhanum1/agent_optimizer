# Axis study v5: predictions

**Status: FROZEN 2026-11-10 (about 21:35 Toronto time), before the main sweep was
submitted.** Seen beforehand:
- the v4 results (instances 1–15), which motivate every hypothesis below;
- a local smoke test of the v5 code (d = 5, four cells, the five cheap solvers,
  plus TuRBO on two cells). Only errors and file formats were checked, not regret;
- a TuRBO timing pilot on Trillium (v4 task keys, instances 16–17), from which only
  wall time and local-data size were read, never regret.

Later edits go in the amendment log at the end, and the original text stays in
place. This file is in `tasks/`, so it must never reach an agent sandbox.

## Setup

- Design: [`axis_v5.py`](axis_v5.py) and `benchmark/AXIS_V5_PLAN.md`.
- Solvers: `sobol` (the reference), `lbfgsb`, `nelder_mead`, `cobyla`, `cma_es`,
  `turbo`. The first five restart from uniform random points (`restart=True`).
  TuRBO is not in `campaign.RESTARTABLE`, so `restart=True` leaves it unchanged.
  It restarts its own trust region when that region shrinks below 0.5⁷.
- Budget: **fixed at 300·d**. The 10/30/100·d checkpoints are kept for description
  only. No claim of the form "fails at a small budget, succeeds at a large one" is
  made or tested.
- Seeds 15–29 = COCO instances 16–30. None were used in v4, so v5 is an
  independent confirmation of v4.
- Groups (d = 5, 20, 40 unless the plan drops d = 40 for cost):

| Group | Functions | β |
|---|---|---|
| plateau | f7 step ellipsoid | 0 |
| noise_moderate | f1, f9, f10 | 0.01 |
| noise_severe | f1, f9, f10 | 1 |
| katsuura | f23 Katsuura | 0 |
| weierstrass | f16 Weierstrass | 0 |
| illcond | f10, f11, f12 | 0 |

## Definitions (as in v4)

- **Score**: log₁₀ regret at 300·d, floored at 1e-8. Regret is the noise-free f
  at the best-observed point, minus f*.
- **Advantage**: the median over seeds of score(Sobol) − score(solver).
- **Fail on a cell**: advantage < 0.1 decades.
- **Collapse on a group** (or a scope, e.g. "weierstrass@20,40" = its d = 20 and
  40 cells): fails on ≥ 80% of the scope's cells.
- **A beats B in a cell**: a paired (by seed) two-sided Wilcoxon signed-rank test,
  Holm-corrected over all 15 solver pairs in the cell, gives p < 0.05, and the
  median of score(B) − score(A) is > 0.
- **A beats B on a scope**: A beats B in ≥ ⅔ of the scope's cells, and B beats A in
  none.
- **Rank reversal (A, B) confirmed**: A beats B on at least one listed scope, and B
  beats A on at least one listed scope.

## Hypotheses: who fails, and who wins there

| | Collapses | On | Who beats it there | Basis |
|---|---|---|---|---|
| H1 | Nelder–Mead | plateau | — | v4: advantage −0.76, last on every seed |
| H2 | L-BFGS-B | noise_moderate | — | v4: collapsed on 100% of cells; Moré & Wild 2009 |
| H3 | COBYLA | noise_severe | — | v4: collapsed on 89% of cells; bbob-noisy, Hansen et al. 2009 |
| H4 | CMA-ES | katsuura | Nelder–Mead, COBYLA | v4, **post hoc**: advantage +0.04/+0.01/−0.01 at d = 5/20/40 |
| H5 | CMA-ES | weierstrass@20,40 | COBYLA | v4, **post hoc**: advantage +0.07/+0.02 at d = 20/40 (+1.07 at d = 5) |
| H6 | TuRBO | illcond | L-BFGS-B | GP length scales and trust regions can't resolve curvature ratios of 10⁶; Le Riche & Picheny 2021, "Revisiting Bayesian optimization in the light of the COCO benchmark" (citation from memory, not checked) |

(v4 advantages for H4 and H5 are differences of medians from v4's `curves.csv`.)

H4 and H5 were found **after** v4 ran: v4 predicted no CMA-ES collapse at all.
v5 is their first test on data that played no part in forming them.

A hypothesis is confirmed in full if the collapse holds and every listed winner
beats the failing solver on that scope.

A note written before any result: H6 is the riskiest prediction. TuRBO is a local
method, so it may well improve on Sobol by more than 0.1 decades on f10–f12 (in
v4, every solver did on most illcond cells) and still lose clearly to
L-BFGS-B. If so, H6's collapse fails while its "L-BFGS-B beats TuRBO" part holds.

## Predicted rank reversals

| | Pair | A beats B on | B beats A on | Basis |
|---|---|---|---|---|
| R1 | L-BFGS-B vs TuRBO | illcond | noise_moderate, noise_severe | H6 plus H2; a GP with a fitted noise term averages noise, finite differences don't |
| R2 | CMA-ES vs TuRBO | illcond | katsuura, weierstrass@20,40 | H6 plus H4/H5; CMA-ES learns the covariance (v4: 1–9 decades ahead of Sobol on illcond) |
| R3 | Nelder–Mead vs CMA-ES | katsuura | plateau | H4; v4 plateau: CMA-ES beat Nelder–Mead 3–0 |
| R4 | COBYLA vs CMA-ES | katsuura, weierstrass@20,40 | noise_severe, plateau | H4/H5; v4: CMA-ES beat COBYLA 9–0 (noise_severe) and 3–0 (plateau) |
| R5 | L-BFGS-B vs CMA-ES | illcond | noise_moderate, noise_severe | v4 R1, replication |
| R6 | L-BFGS-B vs COBYLA | illcond | noise_moderate | v4 R2, replication |
| R7 | L-BFGS-B vs Sobol | illcond | noise_moderate | v4 R3, replication |
| R8 | Nelder–Mead vs Sobol | illcond | plateau | v4 R4 (illcond side) and v4 plateau (Sobol beat Nelder–Mead 3–0) |

R1 and R2 are the TuRBO reversals. Their TuRBO side (TuRBO beats L-BFGS-B under
noise, and beats CMA-ES on katsuura and weierstrass) is new and untested. Of the two,
the Katsuura/Weierstrass side of R2 is the less certain: CMA-ES failing there does
not mean TuRBO succeeds there.

## Secondary predictions (replications of v4, and TuRBO)

| Solver | Collapses on | Does not collapse on |
|---|---|---|
| L-BFGS-B | noise_severe, plateau, katsuura, weierstrass | illcond |
| Nelder–Mead | noise_moderate, noise_severe | illcond |
| COBYLA | — | illcond, noise_moderate, plateau |
| CMA-ES | (H4, H5 only) | illcond, noise_moderate, noise_severe, plateau |
| TuRBO | (H6 only) | noise_moderate |

The only TuRBO prediction beyond H6, R1 and R2 is that it does not collapse on
noise_moderate. Its behaviour on plateau, noise_severe, katsuura and weierstrass is
left open.

## Success criterion for the suite

- Every non-reference solver collapses on at least one group (H1–H6 all hold, at
  least on their collapse part).
- R1 and R2 are both confirmed: TuRBO takes part in rank reversals.
- At least 5 of R1–R8 are confirmed.

## Amendment log

### Amendment 1 (2026-11-10, ~21:38, before the main sweep was submitted; no v5 result seen)

**d = 40 is dropped for every solver and task.** The TuRBO timing pilot showed that
its local data grows without restarts at d = 40 on illcond and plateau, and that one
d = 40 campaign would take ≥ 19 h (see the plan). The budget stays 300·d.
Consequences, decided before any result:
- every group has 2 cells per function (d = 5, 20);
- "weierstrass@20,40" in H5, R2 and R4 now means the single d = 20 cell, so on
  that scope a collapse or a win needs that one cell;
- nothing else changes.
