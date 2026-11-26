# Decisions

## Block 1 (after 0 of 250 measurements)

**Budget:** 0 used / 250 total; 250 remain.
**Question:** Can a GP on public descriptors of binary mixtures (350-dim pool) guide toward low viscosity?
**Mode:** both.
**Gain:** random init + BO gives first landscape read.
**Evidence:** Pool of candidates, descriptors of two components plus mole fractions; low viscosity expected for small/light molecules. No measurements yet.
**Decision:** gp_bo logei, restart.
**Hyperparameters:** n_init 15 (~6% budget), seed 0.
**Block length:** 40, then review progress versus random baseline.
**Review:** improvement over init best, GP predictive quality, consider turbo or log-transform.

Proposal: `{"script": "lib/gp_bo.py", "label": "gp", "args": {"seed": 0, "n_init": 15, "acquisition": "logei"}, "n_points": 40, "state": "restart", "warm_start": false}`

## Note after 15 measurements

The optimizer of block 1 (gp) crashed; its state was not saved, so a later block for this label must restart.

```
PermissionError: [Errno 13] Permission denied: '/proc/self/maps'
```


Runner: Block finished: 15 paid measurements; 235 remain.

## Block 2 (after 15 of 250 measurements)

**Budget:** 15 used / 250 total; 235 remain.
**Question:** Can GP-BO on the 5595-candidate pool reduce viscosity (best so far 0.611, random init mean 3.98)? Block 1 crashed in the GP fit.
**Mode:** exploit.
**Gain:** model-guided search from 15 random observations.
**Evidence:** Crash reproduced offline: botorch's threadpoolctl reads /proc/self/maps (permission denied). work/gpfix.py is gp_bo with that call replaced by a no-op; it passes check.py with warm start.
**Decision:** restart gp_bo wrapper, warm-started on the 15 measurements (no new init needed since n_init=10 <= 15).
**Hyperparameters:** logei default, seed 1.
**Block length:** 40, to assess progress against random baseline.
**Review:** improvement rate, whether y heavy-tailed (consider log transform), consider turbo.

Proposal: `{"script": "work/gpfix.py", "label": "gpfix", "args": {"seed": 1, "n_init": 10}, "n_points": 40, "state": "restart", "warm_start": true}`


Runner: Block finished: 40 paid measurements; 195 remain.

## Block 3 (after 55 of 250 measurements)

**Budget:** 55 used / 250 total; 195 remain.
**Question:** Is GP-BO still improving (best 0.611 -> 0.526 over 40 points)?
**Mode:** exploit.
**Gain:** continued model-guided search; many recent points are 0.6-0.8, showing the model concentrates in a good region.
**Evidence:** y heavy-tailed; the gpfix block found 0.526 and many values below 0.9. No better alternative tested offline.
**Decision:** continue gpfix state unchanged.
**Hyperparameters:** unchanged (logei, seed 1).
**Block length:** 50, then review rate and consider TuRBO/log transform.
**Review:** improvement rate vs. the last block.

Proposal: `{"script": "work/gpfix.py", "label": "gpfix", "args": {"seed": 1, "n_init": 10}, "n_points": 50, "state": "continue", "warm_start": false}`


Runner: Block finished: 50 paid measurements; 145 remain.

## Block 4 (after 105 of 250 measurements)

**Budget:** 105 used / 250 total; 145 remain.
**Question:** Is GP-BO still improving? Best 0.526 -> 0.477 in block 3.
**Mode:** exploit.
**Gain:** further gradual improvement.
**Evidence:** Best improved each block (0.611, 0.526, 0.477); no alternative tested.
**Decision:** continue gpfix state unchanged.
**Hyperparameters:** unchanged (logei, seed 1).
**Block length:** 50, then review.
**Review:** improvement rate; consider TuRBO if stalled.

Proposal: `{"script": "work/gpfix.py", "label": "gpfix", "args": {"seed": 1, "n_init": 10}, "n_points": 50, "state": "continue", "warm_start": false}`


Runner: Block finished: 50 paid measurements; 95 remain.

## Block 5 (after 155 of 250 measurements)

**Budget:** 155 used / 250 total; 95 remain.
**Question:** Is GP-BO still improving? Block 4 best 0.508 vs incumbent 0.477 (no new best).
**Mode:** exploit.
**Gain:** further gradual improvement near the 0.48-0.51 region.
**Evidence:** block minima 0.526, 0.477, 0.508; slowing but still finding values near the best; no alternative tested offline.
**Decision:** continue gpfix state unchanged.
**Hyperparameters:** unchanged (logei, seed 1).
**Block length:** 45, then review the final ~50.
**Review:** if there is still no new best, switch to local refinement or TuRBO around the best.

Proposal: `{"script": "work/gpfix.py", "label": "gpfix", "args": {"seed": 1, "n_init": 10}, "n_points": 45, "state": "continue", "warm_start": false}`


Runner: Block finished: 45 paid measurements; 50 remain.

## Block 6 (after 200 of 250 measurements)

**Budget:** 200 used / 250 total; 50 remain.
**Question:** Can a local trust region around the best beat the stalled global GP-BO?
**Mode:** exploit.
**Gain:** small improvement below 0.477.
**Evidence:** block minima 0.611, 0.526, 0.477, 0.508, 0.499: no new best in two blocks.
**Decision:** restart with TuRBO, warm-started from all 200 observations (counts toward init, so no new design cost).
**Hyperparameters:** library defaults, seed 2.
**Block length:** 50, the rest of the budget.
**Review:** final.

Proposal: `{"script": "lib/turbo.py", "label": "turbo", "args": {"seed": 2}, "n_points": 50, "state": "restart", "warm_start": true}`

## Note after 200 measurements

The optimizer of block 6 (turbo) crashed; its state was not saved, so a later block for this label must restart.

```
PermissionError: [Errno 13] Permission denied: '/proc/self/maps'
```


Runner: Block finished: 0 paid measurements; 50 remain.

## Block 7 (after 200 of 250 measurements)

**Budget:** 200 used / 250 total; 50 remain.
**Question:** Final block; turbo crashed (permission error) with no measurements.
**Mode:** exploit.
**Gain:** small improvement below 0.477.
**Evidence:** gpfix is the only method that has run cleanly; it produced the best point; its saved state is intact.
**Decision:** continue gpfix (logei) rather than retry the crashing turbo.
**Hyperparameters:** inherited, unchanged.
**Block length:** 50, the remaining budget.
**Review:** final.

Proposal: `{"script": "work/gpfix.py", "label": "gpfix", "args": {"seed": 1, "n_init": 10}, "n_points": 50, "state": "continue", "warm_start": false}`


Runner: Block finished: 50 paid measurements; 0 remain.
