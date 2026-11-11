# Axis study v3: plan (draft for approval)

Status: **draft, 2026-11-10. Nothing has run.**

## Goal

Find the problem properties that make each optimizer family fail, using a design
that can actually detect a failure. v1 (L18, budget 10d+50) found one robust
effect: local methods lose on multimodal functions. Its other effects matched the
false-positive rate. Three reasons: the budget was too short for most optimizers
to leave random-search territory; the orthogonal array tangled main effects with
the function and with interactions; and local solvers quit early.

## Will this plan work?

Yes, with four changes to the original idea:

1. **Drop SMAC along with GP-BO.** Measured on the L18 runs, SMAC costs 0.3–0.6 s
   per evaluation, about the same as GP-BO (0.15–0.6 s). The cheap optimizers cost
   0.1–8 ms. At the budgets below, SMAC alone would cost more than everything else
   combined. It returns with TuRBO once the benchmark is fixed.
2. **Restart local solvers.** On L18, COBYLA stopped early in 92% of runs, BOBYQA
   in 44% and L-BFGS-B in 22%. BOBYQA quits after 62 evaluations on a 20-d
   convex quadratic. A bigger budget is useless to a solver that stops, so it
   would "fail" by quitting, not because of the landscape. COCO practice is
   independent restarts from uniform random points until the budget is spent.
   The first start is random too, not the box centre: the pilot found a
   centre-start artifact.
3. **Choose different axes from the earlier four.** Variable type and sparsity
   have little literature for these optimizers (see "Axes not chosen").
   Conditioning and modality do, and they are the axes COCO itself is built on.
4. **Change the trace format.** At 300·d with full per-evaluation traces, the
   sweep writes about 125 M rows (tens of GB). Instead, log best-so-far
   improvements plus fixed checkpoints, as COCO does.

## Baselines

| Key | Family | Restarts |
|---|---|---|
| `random_search` | Reference | n/a |
| `sobol` | Reference (space-filling) | n/a |
| `lbfgsb` | Local, quasi-Newton (finite-difference gradients) | uniform random |
| `nelder_mead` | Local, direct search (simplex) | uniform random |
| `bobyqa` | Local, quadratic-model trust region | uniform random |
| `cobyla` | Local, linear-model trust region | uniform random |
| `cma_es` | Evolution strategy | uniform random, same population (not IPOP/BIPOP) |
| `de` | Differential evolution (NP = 10·d) | none needed |

Deferred: `turbo` (after the benchmark is frozen), `smac`. Dropped: `gp_bo`.

Every restart uses the optimizer's default settings and a fresh uniform start,
so restarts are the same for every family and not tuned for any one of them.

## Axes and levels

Full factorial over the three non-function axes (27 cells) × 9 functions (3 per
structure level) = 243 problems, each on 10 instances (seeds pick the BBOB
instance, rotation and noise stream). All functions are BBOB functions from
`ioh`, built with COCO's own transformations.

| Axis | Low | Medium | High |
|---|---|---|---|
| **Global structure** (3 shapes per level) | unimodal: f1 sphere, f8 Rosenbrock, f13 sharp ridge | multimodal, adequate structure: f15 Rastrigin, f17 Schaffer F7, f19 Griewank–Rosenbrock | multimodal, weak structure: f20 Schwefel, f21 Gallagher 101, f24 Lunacek bi-Rastrigin |
| **Conditioning** (added rotated Λ, as COCO builds f10 from f1 and f18 from f17) | 1 | 10³ | 10⁶ |
| **Dimension** (COCO standard dimensions) | 5 | 20 | 40 |
| **Noise** (COCO `bbob-noisy` Gaussian model, f·exp(β·N(0,1)), reimplemented with a seedable generator) | β = 0 | β = 0.01 (COCO "moderate") | β = 1 (COCO "severe") |

Notes on the choices:
- Each structure level has three shapes that differ in a way the literature
  says matters:
  - Unimodal:
    - sphere (smooth, convex);
    - Rosenbrock (a curved valley, non-convex);
    - sharp ridge (non-smooth). L-BFGS-B's finite-difference gradients and
      BOBYQA's quadratic models assume smoothness.
  - Adequate structure:
    - Rastrigin (regular cosine grid);
    - Schaffer F7 (irregular, high-frequency);
    - Griewank–Rosenbrock (a Rosenbrock-like trend with ripples).
  - Weak structure:
    - Schwefel (deceptive: the second-best region is far from the best);
    - Gallagher 101 (random peaks);
    - Lunacek bi-Rastrigin (two funnels, the wider one wrong). Pilot showed it
      separates optimizers.

  The ellipsoids (f2, f10–f12) are left out because conditioning already covers
  them: sphere at 10⁶ is an ellipsoid.
- Sphere at conditioning 10⁶ is essentially BBOB's f10 ellipsoid, and Schaffer
  F7 at 10³ is f18. So the corners are standard problems people already use.
- Katsuura and Weierstrass are left out because nobody made progress on them in
  the pilot.
- The noise is multiplicative, as in `bbob-noisy`, not the fixed-σ additive noise
  of v1. v1's severe additive noise, and the pilot's heteroscedastic noise, made
  every optimizer look the same. COCO's model is the one the noisy-optimization
  literature reports on. COCO's own C noise generator can't be seeded (see
  `coco_noisy.py`), so the formula is reimplemented.
- Rotation is always on (except where it changes nothing, as on the sphere). So
  separability is not an axis, and DE's crossover can't exploit it.

## Literature behind each axis

Citations are from memory. Step 1 below checks each one for exact title, venue
and the specific claim, before predictions are frozen.

| Axis | Prediction | Sources |
|---|---|---|
| Global structure | Local solvers win unimodal and lose multimodal. CMA-ES with restarts copes with adequate structure. Everyone struggles on weak structure, where DE and random search lose least. | Hansen et al. 2009, BBOB function definitions (INRIA RR-6829); Hansen et al. 2010, "Comparing results of 31 algorithms from BBOB-2009"; Pošík, Huyer & Pál 2012, *Evolutionary Computation*; Rios & Sahinidis 2013, *J. Global Optim.* |
| Conditioning | CMA-ES is close to invariant (it learns the covariance). L-BFGS-B and BOBYQA cope on unimodal problems. Nelder–Mead, DE and random search degrade. | Hansen & Ostermeier 2001, *Evolutionary Computation*; Hansen et al. 2011, "Impacts of invariance in search", *Applied Soft Computing*; Salomon 1996, *BioSystems* (rotation) |
| Noise | L-BFGS-B's finite-difference gradients fail first. Nelder–Mead stalls, and BOBYQA's model is corrupted. Population methods (CMA-ES, DE) are the most robust. | Arnold & Beyer 2003, *Comput. Optim. Appl.* (ES versus direct search under noise); Moré & Wild 2009, *SIAM J. Optim.* (noisy DFO benchmark); Hansen et al. 2009, bbob-noisy definitions (RR-6869) |
| Dimension | Random and Sobol lose ground fastest. Nelder–Mead degrades sharply past about 10. CMA-ES and the model-based local solvers scale best on unimodal problems. | Gao & Han 2012, *Comput. Optim. Appl.* (Nelder–Mead in high dimension); Torczon 1989 (thesis); Rios & Sahinidis 2013; Varelas et al. 2018, bbob-largescale (PPSN) |

These go into a new frozen predictions file before any run, like
`tasks/PREREGISTERED_PREDICTIONS.md`.

### Axes not chosen (for now)

- **Variable type.** The four local solvers have no native way to handle integer
  or categorical variables. Here the axis would measure the rounding wrapper, not
  the optimizer. The literature (bbob-mixint, Tušar et al. 2019; SMAC, Hutter
  et al. 2011) is about methods built for mixed spaces. It returns with SMAC and
  TuRBO.
- **Sparsity (effective dimension).** The literature is about Bayesian
  optimization (REMBO, Wang et al. 2016) and random search (Bergstra & Bengio
  2012). For the cheap set it would mostly measure random search. It returns with
  TuRBO.

## Budget

- **Fixed: 300·d** evaluations per run (1,500 at d = 5, 6,000 at d = 20, 12,000
  at d = 40). Each run is also scored at checkpoints 10·d, 30·d and 100·d, read
  off the same run, so effects can be seen growing with budget.
- The pilot (step 3) checks that 300·d is enough. The checks:
  - (a) on noiseless low-conditioning unimodal cells, the best optimizer reaches
    regret ≤ 1e-8;
  - (b) in at least half the cells, the best optimizer is at least 1 decade
    ahead of random search.

  If either check fails, that is reported before the sweep, and the budget is
  not changed silently.

## Metric and analysis

- **Score:** log₁₀ regret of the noise-free f at the best-*observed* point (as
  now), floored at 1e-8. Report it relative to random search's median in the
  same cell and function, plus each optimizer's rank within the cell.
- **Model,** at each checkpoint: score ~ optimizer × (structure + conditioning +
  dimension + noise) + function nested in structure. Instances are replicates.
  With 3 functions per level, the structure effect can be told apart from
  function-to-function spread within a level.
  The full factorial gives clean main effects and all two-way interactions.
- **An axis "differentiates"** if both hold:
  1. its optimizer × axis interaction is significant: a permutation test with
     levels permuted within function blocks, Holm-corrected over 4 axes ×
     checkpoints;
  2. at least one pair of optimizers swaps order between low and high, by
     ≥ 0.5 decades.
- **Per-family profile:** for each optimizer, its relative score at each level
  of each axis, averaged over the other three. One plot per family.
- **Calibration:** random search's effect against its own median must come out
  null. On L18 it did not: there was a spurious variable-type "effect". So this
  check is a guard against false positives.
- **Robustness:** leave one function out at a time, and report effects that
  survive in each function separately.

## Steps

1. **Verify citations** (no code). Check each source above for exact reference
   and claim. Drop or replace any that doesn't hold. Then freeze the predictions
   file.
2. **Engineering** (benchmark only, no change to the skill):
   - new task module `tasks/axis_v3.py` with the 162 problems;
   - the COCO-style noise wrapper;
   - a restart wrapper for the local solvers and CMA-ES, with random starts;
   - checkpoint- and improvement-only trace logging.

   The `turbo.py` change (q = 1, success tolerance 3) is already committed, as
   a607013 on branch `axis-v3`.

   Tests:
   - f_opt is exact at every level;
   - noise is seeded and reproducible;
   - restarts use the whole budget;
   - checkpoints match a full trace on a few runs.
3. **Pilot:** the corner cells (low/high on conditioning, dimension and noise)
   on all 6 low- and high-structure functions, with 5 instances and 8
   optimizers, to 300·d. That is 1,920 runs and about 25 CPU-h, mostly BOBYQA
   at d = 40 (see "Compute").
   Check the budget checks and that no noise level makes every optimizer look
   the same. If one does, adjust *that level only*, and record the change
   before the sweep.
4. **Full sweep:** 243 problems × 10 instances × 8 optimizers = 19,440 runs.
   Estimated about 180 CPU-h (see "Compute"). Runs to `axis_v3/` under the
   run root, and can be resumed.

## Compute

Costs are per evaluation, measured on the M1 Pro, including the BBOB evaluation
and the run harness:

| Optimizer | d = 5 | d = 20 | d = 40 |
|---|---|---|---|
| random, Sobol, L-BFGS-B, Nelder–Mead, DE, CMA-ES | 0.06–0.16 ms | 0.2–0.3 ms | about 0.3–0.45 ms (extrapolated) |
| COBYLA | 1.0 ms | 1.35 ms | 1.5 ms |
| BOBYQA (Py-BOBYQA) | 1.6 ms | 12.8 ms | 55 ms |

Each optimizer runs 810 times per dimension, for 15.8 M evaluations per
optimizer. In M1 Pro core-hours:
- the six cheap optimizers: about 7 h;
- COBYLA: about 6 h;
- BOBYQA: about 165 h, of which about 150 h is d = 40.

So the total is **about 180 core-hours**, and BOBYQA is 90% of it. A single
BOBYQA run at d = 40 takes about 11 minutes.

Single-core speed, measured with the same test on each machine (M1 Pro = 1.0):
work_machine 1.5×, Trillium login node 1.33×. Trillium's login node was
loaded, so its compute nodes are probably at least that fast.

| Machine | Usable cores | Pilot (about 25 core-h) | Sweep (about 180 core-h) |
|---|---|---|---|
| Local M1 Pro (6 performance + 2 efficiency cores, 16 GB) | about 7 | about 3–4 h | about 25–30 h |
| work_machine (i7-13700K, 8 performance + 8 efficiency cores, 24 threads, 30 GB, idle) | about 20 core-equivalents | about 1.5 h | about 9–10 h |
| Trillium (1 compute node: 2 × EPYC 9655, 192 cores; account `def-moosavi5`, 24 h limit) | 192 × 1.33 | about 10 min + queue | about 1 h + queue (bounded by the 8–11 min BOBYQA d = 40 runs) |

Notes:
- **BLAS threads.** Set `OMP_NUM_THREADS=1` (and `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`) on every machine. Otherwise numpy threads oversubscribe
  the cores.
- **Trillium setup.**
  - Clone and `uv sync` on a login node; compute nodes have no internet.
  - Write to `$SCRATCH`. The sweep writes about 40 k small files, so pack them
    per task at the end.
  - Trillium allocates whole nodes, so one node runs the full sweep as a single
    job. Queue wait on a `def-` account is unknown.
- **Option to cut cost:** replace Py-BOBYQA with NLopt's BOBYQA, which is a C
  translation of Powell's code. Then the sweep would be about 20 core-hours
  everywhere. It changes the baseline's implementation, and Py-BOBYQA's
  stopping behaviour is also the one that quit after 62 evaluations.
5. **Analysis** and a short report: which axes differentiate, the per-family
   profiles, and predictions scored against the frozen file.
6. **Later, separately:** add TuRBO (and SMAC if wanted) on a subset of cells
   chosen from step 5. Then choose the agent's cells.

## Decisions (2026-11-10)

1. SMAC is excluded from the cheap sweep.
2. Dimensions are 5/20/40.
3. Noise is COCO's multiplicative Gaussian model, β = 0 / 0.01 / 1.
4. CMA-ES restarts with the same settings (no IPOP/BIPOP).
5. Budget is 300·d.
6. There are three shapes per structure level, listed above.
