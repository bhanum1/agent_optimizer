# Where each solver collapses

A working document: for each baseline, the problem property that **collapses**
it, either observed in the axis study v3 sweep or predicted from the
literature and not yet tested. The goal is one collapse axis per optimizer
family, so the benchmark can tell the families apart.

The data is the v3 sweep: 17,010 campaigns, 300·d budget, local solvers and
CMA-ES restarted from random points. The design is in
[AXIS_STUDY_PLAN.md](AXIS_STUDY_PLAN.md) and the frozen predictions in
`src/aobench/tasks/AXIS_V3_PREDICTIONS.md`. The run outputs are in
`runs/axis_v3/` (not in git).

## Definition of collapse

- **Advantage**: how far a solver is ahead of random search on one task, in
  decades of regret. It is the median over 10 seeds of
  log₁₀ regret(random search) − log₁₀ regret(solver), at 300·d.
- **Collapsed on a task**: advantage < 0.1 decades. The solver is no better
  than sampling uniformly.
- **Collapses on an axis level**: at that level, at least 80% of the solver's
  tasks are collapsed, against at most about 40% at the axis's easy level.

This replaces the relative score in `axis_v3_analyze.py` for this question. The
relative score compares a solver with the mean of all solvers, so the reference
methods appear to "improve" whenever the others get worse. Advantage over
random search does not have that problem.

## Summary

| Solver | Family | Collapse axis | Status | Advantage, easy → collapsed (decades) | Share of tasks collapsed, easy → collapsed |
|---|---|---|---|---|---|
| L-BFGS-B | local, gradient (finite differences) | **noise**, already at moderate (β = 0.01) | observed | 3.07 → −0.63 | 15% → 100% |
| Nelder–Mead | local, direct search | **noise** (moderate); **rugged multimodality** | observed | 1.59 → −0.66 (noise); 0.57 → −0.43 (shape) | 40% → 95% (both) |
| COBYLA | local, linear-model trust region | **severe noise** (β = 1) | observed | 2.07 → −0.11 | 22% → 88% |
| DE | population, evolutionary | none: it barely beats random anywhere at 300·d | budget-limited | 0.02–0.38 everywhere | 27% (d = 5) → 80% (d = 40) |
| CMA-ES | population, evolution strategy | **none found** | needs a new axis | weakest: rugged × severe noise, 0.22 | at most 33% in any 2-axis cell |
| random, Sobol | reference | not applicable | — | Sobol − random ≈ 0 on every axis (calibration passes) | — |

## Per solver

### L-BFGS-B: noise

- **Noise.** L-BFGS-B is one of the best solvers without noise (mean rank 2.5,
  3.07 decades ahead of random). With even moderate multiplicative noise
  (β = 0.01, a 1% relative error), it is worse than random on 100% of tasks.
  The cause is its finite-difference gradients: the noise is divided by the
  step size, so the gradient is noise. Prediction P3 is confirmed.
- **Rugged multimodality** (Rastrigin, Schaffer F7, Griewank–Rosenbrock,
  Lunacek): 78% of tasks collapsed. This is secondary: restarts keep it partly
  alive.
- **Not tested yet:** non-smoothness. Sharp ridge did not collapse it (0.88
  decades). Plateaus (step functions) are the classic case where finite
  differences return a zero gradient.

### Nelder–Mead: noise and rugged multimodality

- Collapses at moderate noise (95% of tasks) and on rugged multimodal shapes
  (95%).
- It is weak everywhere. With restarts it never restarts (median 0 restarts):
  in 20–40 dimensions the simplex stalls without ever meeting its own
  tolerances. So it often runs one stalled simplex for the whole budget.
- The dimension effect predicted by the literature (Torczon 1989; Gao & Han
  2012) does not show, because it is already near random at d = 5.
- **To check:** the stall. Restarting on stagnation (no improvement in k·d
  evaluations) is a common fix. It would change the baseline, so it is a
  decision, not a bug fix.

### COBYLA: severe noise

- COBYLA survives moderate noise (1.39 decades ahead). Its trust-region radius
  starts at 0.1 of the box, so its linear models average over the noise.
- At β = 1 it collapses on 88% of tasks.
- Rugged shapes are its secondary weakness (58% collapsed).
- **Not tested yet:** strong *expanding* ill-conditioning. Its linear models
  carry no curvature (prediction P2). This needs the conditioning fix below.

### DE: budget, not an axis

- DE (NP = 10·d, F = 0.8, CR = 0.9) gets 30 generations at 300·d in every
  dimension. Its advantage over random is at most 0.38 decades anywhere
  (0.02–0.08 at d ≥ 20, at every checkpoint).
- So the data says "DE needs more than 300·d", not "DE collapses on axis X".
- **Its literature axis is untested:** separability. DE's crossover exploits
  axis-aligned structure, and rotation removes that (Salomon 1996). Every v3
  task is rotated, so DE never had separable problems to be good at.

### CMA-ES: no collapse axis yet

- CMA-ES is first on all 9 shapes (mean rank 1.2–2.2) and stays well ahead of
  random on every single axis level (advantage 0.84 or more).
- Its worst cells are interactions with rugged shapes:
  - rugged × high conditioning: 0.44 decades, 33% collapsed;
  - rugged × severe noise: 0.22 decades, 31% collapsed;
  - rugged × d = 40: 0.61 decades, 28% collapsed.

  These are weaknesses, not collapses.
- The literature points to three properties v3 did not vary (next section).

## Axes still to define

These candidates are chosen so that each family without a collapse axis gets
one predicted from the literature:

| Candidate axis | Target | Mechanism | Sources (from memory, to be checked) | How to build it |
|---|---|---|---|---|
| **Plateaus / discreteness** | CMA-ES, L-BFGS-B | On flat steps, CMA-ES's step size shrinks below the step width and stalls. Finite-difference gradients are zero. | Hansen 2011, "A CMA-ES for mixed-integer nonlinear optimization" (INRIA RR-7751); Hamano et al. 2022, "CMA-ES with margin" (GECCO); BBOB f7 step ellipsoid; the pilot found step ellipsoid separated optimizers | Round the input to a grid of width w (0 / 0.1 / 1) before the base function, so the plateaus grow. Or use integer variables. |
| **Separability (rotation)** | DE, coordinate methods | DE's binomial crossover is coordinate-wise. It is fast on separable problems and slow when they are rotated. | Salomon 1996, *BioSystems*; Hansen et al. 2011, *Applied Soft Computing* | Rotation on or off, with the same shape and conditioning. |
| **Budget relative to d²** | CMA-ES | Covariance learning needs about d² evaluations, so CMA-ES is near random at small budgets (v3: 0.1–0.2 decades at 10·d). | Hansen 2016, "The CMA evolution strategy: a tutorial" | Already in the data as checkpoints (10, 30, 100, 300·d). Report it as an axis rather than run more. |
| **Conditioning (fixed)** | COBYLA, Nelder–Mead, DE | Curvature ratios of 10⁶ in steep directions. | Hansen et al. 2009 (BBOB definitions) | Replace v3's shrinking transform with COCO's stretching Λ. See the bug below. |
| **Ruggedness** (replaces "global structure") | local solvers | Many local optima *everywhere*, not just "multimodal". | Exploratory: v3's data | Rugged (Rastrigin, Schaffer F7, Griewank–Rosenbrock, Lunacek) against few basins (Gallagher 101, Schwefel) against unimodal. **Defined after seeing the data**, so it must be confirmed on new functions before it counts. |

## Known problems with v3

- **The conditioning transform was backwards.** It shrank directions toward
  the optimum, so high conditioning meant fewer directions that matter, and
  every solver *improved*. For example, random search's median log regret was
  2.18, 1.16 and 0.77 at conditioning 1, 10³ and 10⁶. So prediction P2 is
  untested. The L18 tasks used the same construction.
- **BBOB's "weak global structure" group is not harder here.** Gallagher 101
  and Schwefel are among the easiest shapes for CMA-ES (5.40 and 2.52 decades).
  Lunacek behaves like the rugged group. The structure axis as designed does not
  order difficulty, and that is why it showed no effect.
- **Budget check (a) failed.** Only 4 of 9 easy cells (unimodal, conditioning 1,
  noiseless) reached 1e-8 at 300·d.
- **Nelder–Mead never restarts,** and **DE is budget-limited** (see above).
