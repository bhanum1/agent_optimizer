# Decisions

## Block 1 (after 0 of 250 measurements)

**Budget:** 0 used / 250 total; 250 remain.
**Question:** Can GP-BO on the 350-dim descriptor pool learn a useful trend (low-viscosity mixtures) quickly?
**Mode:** both.
**Gain:** random pool design of 20 shows the y distribution; 20 BO steps test model-guided gain.
**Evidence:** Indexed pool of binary mixtures with descriptors; prior: small, light molecules/low-viscosity components likely give low y. No data yet.
**Decision:** gp_bo, restart, logei.
**Hyperparameters:** n_init=20 (8% of budget), defaults otherwise.
**Block length:** 40, then review the fit and progress versus random.
**Review:** best vs initial design, whether BO beats random, consider a log-y transform or other methods.

Proposal: `{"script": "lib/gp_bo.py", "label": "gp1", "args": {"seed": 0, "n_init": 20, "acquisition": "logei"}, "n_points": 40, "state": "restart", "warm_start": false}`

## Note after 20 measurements

The optimizer of block 1 (gp1) crashed; its state was not saved, so a later block for this label must restart.

```
PermissionError: [Errno 13] Permission denied: '/proc/self/maps'
```


Runner: Block finished: 20 paid measurements; 230 remain.

## Block 2 (after 20 of 250 measurements)

**Budget:** 20 used / 250 total; 230 remain.
**Question:** Does GP-BO on the descriptor pool beat random (best of 20 random: 0.611; median 2.0)?
**Mode:** both.
**Gain:** model-guided search for low viscosity from the 20 random points.
**Evidence:** Block 1 gp1 crashed after the 20 init points: botorch's threadpoolctl reads /proc/self/maps, which is denied. A wrapper (work/gp_safe.py) that no-ops threadpool_limits passes check.py with warm start. y is skewed (0.61-16.7).
**Decision:** restart with warm_start of the 20 existing points, using the patched gp_bo (logei).
**Hyperparameters:** library defaults; n_init=20 is met by the warm start so no new design cost.
**Block length:** 40, then review progress versus random.
**Review:** improvement rate, GP predictive quality (LOO), consider turbo or a log transform.

Proposal: `{"script": "work/gp_safe.py", "label": "gp2", "args": {"seed": 1, "n_init": 20, "acquisition": "logei"}, "n_points": 40, "state": "restart", "warm_start": true}`


Runner: Block finished: 40 paid measurements; 190 remain.

## Block 3 (after 60 of 250 measurements)

**Budget:** 60 used / 250 total; 190 remain.
**Question:** GP-BO (40 pts) never beat best random 0.611 (BO values median ~1.3 vs 2.0 random). Does a trust-region method do better in 350-dim?
**Mode:** exploit.
**Gain:** local refinement around best regions.
**Evidence:** best stays 0.611 over 40 BO steps; global GP in 350-d is weak.
**Decision:** switch to TuRBO (patched for threadpool), warm start with all 60 points.
**Hyperparameters:** defaults.
**Block length:** 40, then compare with gp2 rate.
**Review:** best vs 0.611; if no gain consider other approaches (feature subset, continuing GP).

Proposal: `{"script": "work/turbo_safe.py", "label": "tr1", "args": {"seed": 2}, "n_points": 40, "state": "restart", "warm_start": true}`


Runner: Block finished: 40 paid measurements; 150 remain.

## Block 4 (after 100 of 250 measurements)

**Budget:** 100 used / 250; 150 remain.
**Question:** Is TuRBO still improving?
**Mode:** exploit.
**Gain:** TuRBO block: median 0.64 vs 1.3 (GP) and 2.0 (random); best 0.611 -> 0.477.
**Evidence:** per-20 block stats from work/a.py.
**Decision:** continue tr1 (saved state).
**Hyperparameters:** unchanged.
**Block length:** 50, then review.
**Review:** best progress, trust region collapse/restarts.

Proposal: `{"script": "work/turbo_safe.py", "label": "tr1", "args": {"seed": 2}, "n_points": 50, "state": "continue", "warm_start": false}`


Runner: Block finished: 50 paid measurements; 100 remain.

## Block 5 (after 150 of 250 measurements)

**Budget:** 150 used / 250; 100 remain.
**Question:** Is TuRBO still productive?
**Mode:** exploit.
**Gain:** TuRBO medians far better than GP/random; best 0.477 (row 80), top values clustered 0.477-0.53.
**Evidence:** work/a.py stats; no better method identified.
**Decision:** continue tr1.
**Hyperparameters:** unchanged.
**Block length:** 50, then review for the final 50.
**Review:** best progress; trust region restarts.

Proposal: `{"script": "work/turbo_safe.py", "label": "tr1", "args": {"seed": 2}, "n_points": 50, "state": "continue", "warm_start": false}`


Runner: Block finished: 50 paid measurements; 50 remain.

## Block 6 (after 200 of 250 measurements)

**Budget:** 200 used / 250; 50 remain.
**Question:** Is TuRBO still improving?
**Mode:** exploit.
**Gain:** best improved 0.477 -> 0.448 in last block.
**Evidence:** status shows best 0.4484 at row 172.
**Decision:** continue tr1 to finish the budget.
**Hyperparameters:** unchanged.
**Block length:** 50, final.
**Review:** none.

Proposal: `{"script": "work/turbo_safe.py", "label": "tr1", "args": {"seed": 2}, "n_points": 50, "state": "continue", "warm_start": false}`


Runner: Block finished: 50 paid measurements; 0 remain.
