# Axis study v3: predictions

**Status: FROZEN 2026-11-10,** written before any v3 sweep result was seen. The
only v3 runs before this were a 28-campaign smoke test of the code (one seed,
four tasks), whose results were not analysed. Later edits go in an amendment log
at the end, and the original text stays in place.

This file is in `tasks/`, so it must never reach an agent sandbox.

## Setup

- Design: [`axis_v3.py`](axis_v3.py) and `benchmark/AXIS_STUDY_PLAN.md`.
- Baselines: `random_search`, `sobol`, `lbfgsb`, `nelder_mead`, `cobyla`,
  `cma_es`, `de`. Local solvers and CMA-ES restart from uniform random points.
  BOBYQA, GP-BO, SMAC and TuRBO are not in this sweep.
- Budget: 300·d.

## Score

- Score: log₁₀ regret (floored at 1e-8) of the noise-free f at the
  best-observed point, minus random search's median on the same task.
- An optimizer's *level score* on an axis: its mean score at that level,
  averaged over all other axes, functions and seeds.
- "**Degrades**" on an axis: the level score gets worse from low to high by
  ≥ 0.5 decades *relative to the other optimizers*. That is, its within-task
  rank gets worse by ≥ 1 position, and the optimizer × axis interaction is
  significant (Holm-corrected over the 4 axes).
- "**Robust**": neither of those holds.

## Predictions

Sources are from memory. Checking them does not change the predictions; it is
logged in the amendment log.

### P1. Global structure (unimodal → adequate → weak)

- L-BFGS-B, Nelder–Mead and COBYLA degrade. They are the best at unimodal and
  lose their lead on multimodal shapes.
- CMA-ES degrades less than the local solvers, from unimodal to adequate.
- Random search, Sobol and DE improve in rank toward weak structure.

Sources: Hansen et al. 2010 (BBOB-2009 comparison); Pošík, Huyer & Pál 2012;
Rios & Sahinidis 2013.

### P2. Conditioning (1 → 1e3 → 1e6)

- CMA-ES is robust: it adapts its covariance.
- Nelder–Mead and DE degrade.
- L-BFGS-B is robust on unimodal shapes (quasi-Newton).
- COBYLA degrades: its linear models carry no curvature.

Sources: Hansen & Ostermeier 2001; Hansen et al. 2011, "Impacts of invariance
in search"; Salomon 1996.

### P3. Noise (β = 0 → 0.01 → 1)

- L-BFGS-B degrades most: its finite-difference gradients break under noise.
- Nelder–Mead and COBYLA degrade.
- CMA-ES and DE are robust relative to the local solvers.

Sources: Arnold & Beyer 2003; Moré & Wild 2009; Hansen et al. 2009 (bbob-noisy).

### P4. Dimension (5 → 20 → 40)

- Random search and Sobol degrade most.
- Nelder–Mead degrades.
- CMA-ES and L-BFGS-B are robust on unimodal shapes.

Sources: Gao & Han 2012; Torczon 1989; Varelas et al. 2018.

### P0. Calibration

- Random search's own score is 0 by construction. Sobol is robust on every axis.
- Any axis effect "found" for Sobol counts as a false positive.

## Amendment log

(none)
