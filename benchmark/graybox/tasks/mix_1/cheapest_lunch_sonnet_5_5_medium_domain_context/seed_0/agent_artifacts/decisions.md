# Decisions

## Block 1 (after 0 of 250 measurements)

**Budget:** 0 used / 250 total; 250 remain.
**Question:** Can a GP on 350 descriptor features guide search on the candidate pool?
**Mode:** both.
**Gain:** random initial design of 20 then 10 GP-guided points; tests surrogate usefulness.
**Evidence:** Binary-mixture viscosity pool, 350 RDKit-like descriptors; no data yet. Prior only: low-MW components/mixtures tend to have low viscosity.
**Decision:** gp_bo (logei) restart, n_init 20.
**Hyperparameters:** n_init 20 (~8% budget); others default.
**Block length:** 30, to review model quality.
**Review:** compare GP-proposed vs random values; consider turbo or feature subspace.

Proposal: `{"script": "lib/gp_bo.py", "label": "gp", "args": {"seed": 0, "n_init": 20}, "n_points": 30, "state": "restart", "warm_start": false}`

## Note after 20 measurements

The optimizer of block 1 (gp) crashed; its state was not saved, so a later block for this label must restart.

```
PermissionError: [Errno 13] Permission denied: '/proc/self/maps'
```


Runner: Block finished: 20 paid measurements; 230 remain.

## Block 2 (after 20 of 250 measurements)

**Budget:** 20 used / 250 total; 230 remain.
**Question:** Can a surrogate on descriptor features guide the pool search? (gp_bo crashed: botorch fit hits PermissionError on /proc/self/maps via threadpoolctl; reproduced offline.)
**Mode:** exploit.
**Gain:** EI-guided picks should beat the random 20 (best 0.611 cP).
**Evidence:** only 20 random points; prior: low-MW mixtures have low viscosity. Library GP unusable here.
**Decision:** custom numpy GP (Matern52, PCA-20 of standardized pool features, log y, LML grid for lengthscale/noise), EI over unused candidates; warm start on all data.
**Hyperparameters:** npc=20, xi=0; checked with check.py --warm 20.
**Block length:** 40, to judge improvement over random.
**Review:** compare proposed y vs random; adjust npc/exploration.

Proposal: `{"script": "work/npgp.py", "label": "npgp", "args": {"seed": 0, "npc": 20}, "n_points": 40, "state": "restart", "warm_start": true}`


Runner: Block finished: 40 paid measurements; 190 remain.

## Block 3 (after 60 of 250 measurements)

**Budget:** 60 used / 250; 190 remain.
**Question:** Does the GP keep improving the best?
**Mode:** exploit.
**Gain:** npgp block gave best 0.484 (vs 0.611 random best), median 1.79 vs 2.01; modest but positive.
**Evidence:** per-block min/median: random 0.611/2.01, npgp first 20 0.704/1.88, last 20 0.484/1.79.
**Decision:** continue npgp state, no change.
**Hyperparameters:** unchanged (npc=20, xi=0).
**Block length:** 60, then review.
**Review:** improvement rate; consider npc/xi change.

Proposal: `{"script": "work/npgp.py", "label": "npgp", "args": {}, "n_points": 60, "state": "continue", "warm_start": false}`


Runner: Block finished: 60 paid measurements; 130 remain.

## Block 4 (after 120 of 250 measurements)

**Budget:** 120 used / 250 total; 130 remain.
**Question:** Is npgp still improving?
**Mode:** exploit.
**Gain:** block 3 median fell 1.86->1.05 (model concentrating on low region) though best stayed 0.484 (block mins 0.50, 0.51, 0.59).
**Evidence:** per-block stats from work/a.py; median improvement indicates model works; min plateau is plausibly noise of tail.
**Decision:** continue npgp unchanged.
**Hyperparameters:** unchanged (npc=20, xi=0).
**Block length:** 50, then review with 80 left.
**Review:** whether best improves; if not, raise npc/xi or try different kernel.

Proposal: `{"script": "work/npgp.py", "label": "npgp", "args": {}, "n_points": 50, "state": "continue", "warm_start": false}`


Runner: Block finished: 50 paid measurements; 80 remain.

## Block 5 (after 170 of 250 measurements)

**Budget:** 170 used / 250; 80 remain.
**Question:** Is npgp still improving?
**Mode:** exploit.
**Gain:** Block 4 best 0.434 (from 0.484), median 0.96; still improving.
**Evidence:** per-block min/median: 0.611/2.01, 0.484/1.86, 0.50/1.05, 0.434/0.96.
**Decision:** continue npgp unchanged.
**Hyperparameters:** unchanged.
**Block length:** 40, then final review.
**Review:** best change.

Proposal: `{"script": "work/npgp.py", "label": "npgp", "args": {}, "n_points": 40, "state": "continue", "warm_start": false}`


Runner: Block finished: 40 paid measurements; 40 remain.

## Block 6 (after 210 of 250 measurements)

**Budget:** 210 used / 250 total; 40 remaining.
**Question:** Does npgp still find improvements?
**Mode:** exploit.
**Gain:** Block 5 best 0.593 (no improvement over 0.4344), median 1.64; the tail is noisy, though the earlier blocks improved.
**Evidence:** per-block min/median from work/a.py.
**Decision:** continue npgp unchanged for the final 40; no restart cost, and there is no better-supported alternative.
**Hyperparameters:** unchanged.
**Block length:** 40, the whole remaining budget.
**Review:** final.

Proposal: `{"script": "work/npgp.py", "label": "npgp", "args": {}, "n_points": 40, "state": "continue", "warm_start": false}`


Runner: Block finished: 40 paid measurements; 0 remain.
