# Decisions

## Block 1 (after 0 of 250 measurements)

**Budget:** 0 used / 250 total; 250 remaining.
**Question:** Does a domain prior for viscosity (small, non-H-bonding, low-TPSA, acyclic components; log-linear mole-fraction mixing) rank this pool usefully, and how do measured values spread so a surrogate can be built?
**Mode:** both (14 prior-guided exploit picks + 10 random calibration picks).
**Gain:** Likely a low-viscosity incumbent (~0.3-0.5 cP expected if the prior holds) and a first test of the prior's ranking vs random points, deciding whether the next block uses a prior-mean surrogate over engineered features or a more data-driven one.
**Evidence:** No measurements. Pool of 5595 binary mixtures (~266 distinct components, 350 descriptor columns, mole fractions x434/x435). Domain prior (hypothesis, not data): pure viscosity rises with MW, H-bond donors (alcohols, glycols, water), polarity/TPSA, rings; DMSO (~2 cP) penalized explicitly; mixture ln(eta) ~ mole-fraction-weighted. Heuristic score in work/design1.py (prior2). Mixtures of the very lowest-viscosity solvents (ether, pentane) exist only paired with alcohols at <=0.55 fraction.
**Decision:** experiment: fixed_points on 14 top-prior candidates (diverse: each component pair once, each component <=2 times) and 10 seeded random candidates. If prior picks clearly beat random ones, exploit with a prior-informed surrogate; if not, rely on a data-driven surrogate on weighted descriptors.
**Hyperparameters:** fixed_points, seed 1; indices from work/design1.py (rng seed 1 for random picks).
**Block length:** 24 measurements (~10% of budget): enough to calibrate a low-dimensional surrogate and compare prior vs random groups.
**Review:** Compare prior-pick vs random-pick values; correlation of prior2 with log(y); fit a GP/ridge on mole-fraction-weighted features (MW, NHOH, TPSA, rings, logP) and test LOO prediction.

Proposal: `{"script": "lib/fixed_points.py", "label": "init_design", "args": {"seed": 1, "points": [3357, 1037, 3522, 1116, 280, 2915, 1414, 4615, 2085, 5245, 849, 3176, 1409, 935, 1747, 2644, 4602, 194, 5306, 2860, 804, 4221, 5312, 1395]}, "n_points": 24, "state": "restart", "warm_start": false}`


Runner: Block finished: 24 paid measurements; 226 remain.

## Block 2 (after 24 of 250 measurements)

**Budget:** 24 used / 250 total; 226 remaining.
**Question:** Which unmeasured binary mixtures have the lowest viscosity? A structured model (pure-component log-viscosity plus mole-fraction mixing) should be able to rank them, and the question is whether the data can correct its component values fast enough.
**Mode:** exploit (model-guided EI; each measurement also updates the latent viscosities of two components).
**Gain:** Incumbent 0.51 cP (methyl acetate 0.79 + gamma-butyrolactone). The model's top unmeasured predictions are 0.48-0.55 cP with sd about 0.09 log10, so expect a modest gain, perhaps to about 0.45 cP. The block also tests whether the pool has any mixture far below 0.5 cP.
**Evidence:** Block 1: 14 prior picks scored 0.51-1.06 cP (median 0.69); 10 random picks scored 0.71-13.6 cP (median 4.1). The heuristic prior matched the measured log y with Spearman 0.73. Viscosity follows log-linear mixing of pure-component values closely: with literature pure viscosities used as a PRIOR (from domain knowledge, e.g. MeOAc 0.36, GBL 1.7, octane 0.5, decanol 11), predicted log10 y fits within 0.00-0.08 for many points. The pool has 274 distinct components. The very-low-viscosity solvents (ether, pentane, acetone, MeCN, diethylamine, MTBE) occur almost only paired with alcohols, glycols or other viscous partners, so the best pool values are likely about 0.45-0.5 cP rather than about 0.3 cP. Components predicted only by the heuristic were wrong in measured cases: an unknown bicyclic compound (ID 173), adiponitrile, and a compound I take to be N-formylmorpholine (ID 144) all came out more viscous than predicted, and many halogenated components were underpredicted. So recognisable components got knowledge-based pure values (sd 0.08-0.15 log10) and the rest got heuristic values (sd 0.35). Model work/latent_ei.py: log10 y = x1*a[c1] + x2*a[c2] + b + eps, with a GP over component descriptors (RBF) and independent per-component terms. Leave-one-out on the 24 points: RMSE 0.094 log10, z sd 0.85, so it is calibrated or slightly conservative. Maximum likelihood chose sg=0.15, se=0.03.
**Decision:** switch to the custom conjugate-Gaussian latent-component model with expected improvement over unused pool indices, warm-started on all 24 measurements. Its first EI picks are diisopropyl-ether-rich + glycol ether, MTBE-rich + hexanol, methyl acrylate + decanol, and cyclopentane + heptanol.
**Hyperparameters:** prior sds 0.08 (well-known pure values), 0.15 (less certain identifications), 0.35 (heuristic); RBF ell=1.5 on standardized [lnMW, ln(1+NHOH), TPSA, rings, logP, valence electrons]. sg is chosen from {0.1,0.15,0.25} and se from {0.02,0.03,0.05,0.08,0.12} by marginal likelihood at each step; se=0.02 was added because 0.03 sat at the grid edge. Bias b ~ N(0.03, 0.1); xi=0.
**Block length:** 30 measurements. That is enough for EI to work through the predicted-best region and refine component values, and it leaves about 196 for re-planning, for example if the residuals show strong non-ideal mixing (an excess term may be needed).
**Review:** improvement over 0.51; residuals of new points against predictions made before they were measured (does z stay at about 1?); whether se stays at the grid edge; whether EI has drifted into exploring heuristic components; whether a mole-fraction excess term (x1*x2) is warranted.

Proposal: `{"script": "work/latent_ei.py", "label": "latent_ei", "args": {"seed": 0, "se_grid": [0.02, 0.03, 0.05, 0.08, 0.12]}, "n_points": 30, "state": "restart", "warm_start": true}`


Runner: Block finished: 30 paid measurements; 196 remain.

## Block 3 (after 54 of 250 measurements)

**Budget:** 54 used / 250 total; 196 remaining.
**Question:** Is the log-linear latent-component model missing a systematic mixing excess? If it is, EI is ranking the remaining low-viscosity pairs wrongly.
**Mode:** exploit, with some exploration from the model's uncertainty.
**Gain:** Block 2 lowered the best from 0.510 to 0.434 cP (idx 1352: MEK 0.83 + C5 alcohol 0.17). The model predicts the lowest unmeasured pairs at about 0.52–0.56 cP. The probability that each beats 0.434 is roughly 2–7%, and these probabilities sum to about 1 across the pool. So I expect at most a few percent of further gain, plus information on low-η pairs that have not been sampled yet (ether 63, methylal 178, ethyl formate 47, acetone 56 pairs).
**Evidence:** Of the 30 block-2 points, 27 landed in the predicted 0.43–1.0 cP region. Three were large misses: 2458 (water + comp188) at 15.9 cP, 2571 (heavy MW419 comp at 5%) at 4.1 cP, and 4495 (MW423 comp) at 6.1 cP. All three involve unknown or heavy components where the heuristic is poor. Residuals against literature pure viscosities are mostly negative. They are largest (about -0.08 to -0.12 log10) at x1*x2 of 0.13–0.25, especially for pairs where only one component is an H-bond donor (an alcohol in an aprotic solvent, i.e. negative Grunberg-Nissan excess). Adding an excess term x1x2*(g0 + g1*asym) fits g0=-0.23±0.16 and g1=-0.14±0.15. Its LOO log predictive density on the y<1 points is 46.4, against 42.3 for v1, with rmse 0.071 vs 0.075 and z sd 0.86. Domain prior used: literature pure-component viscosities (K0/K1 tables) and the known negative excess of alcohol + aprotic mixtures. Both are treated as priors and updated by the data.
**Decision:** Adjust. Switch to work/latent_ei2.py: the same conjugate latent-component model plus a Gaussian Grunberg-Nissan excess term. EI now uses the predictive sd including se, because pair-specific misfit is real spread in unmeasured y on this deterministic objective. Warm-start from all 54 points.
**Hyperparameters:** sx=0.3 (prior sd of the excess coefficients; sx=0.6 gave the same LOO). se_grid [.02, .03, .05, .08, .12], with ML selecting se=0.03 and sg=0.25. The other settings are unchanged from v1: sk=.08, sk1=.15, su=.35, ell=1.5, b0=.03, sb=.1, xi=0.
**Block length:** 30. EI is sequential and refits every point. Thirty points are enough to work through the high-PI pairs and see whether the excess term predicts better, while leaving 166 for re-planning.
**Review:** Did the best improve below 0.434? Check the z-scores of new points against their pre-measure predictions (opt.hist) and whether g0/g1 are now significant. Watch for EI drifting into unknown heavy components (wasted points). If the model becomes overconfident with EI near 0, consider a fraction-asymmetric excess term (dilute alcohol) or diversity exploration of unknown low-MW aprotic components.

Proposal: `{"script": "work/latent_ei2.py", "label": "latent_ei2", "args": {"seed": 0, "se_grid": [0.02, 0.03, 0.05, 0.08, 0.12], "sx": 0.3, "ei_noise": true}, "n_points": 30, "state": "restart", "warm_start": true}`


Runner: Block finished: 30 paid measurements; 166 remain.

## Block 4 (after 84 of 250 measurements)

**Budget:** 84 used / 250 total; 166 remaining.
**Question:** Can the calibrated latent-component + excess model still find a pair below 0.434 cP, or has the pool's low end been found?
**Mode:** exploit. Each point also refines the latent pure-component values of under-sampled low-viscosity components.
**Gain:** The model gives a small chance of improvement. The top unmeasured pairs are predicted at 0.52–0.53 cP, each with PI ≤ 2%, and the PI sum over the whole pool is about 0.25. Any improvement will come from pairs that turn out lower than predicted. EI ranks those best under a calibrated model.
**Evidence:** Block 3 (30 EI points) did not improve the best of 0.434 cP; its best was 0.484 (idx 5082). The pre-measure z-scores for block 3 have mean 0.01 and sd 1.00, so the model is well calibrated. The largest misses were 4022 (+3.4, a MW-164 OH component) and 1770 (+2.2, ether + C5 alcohol, closer to ideal mixing than the excess term predicted). Leave-one-out over all 84 points, restricted to pairs predicted at log10 y < -0.2 (n=35): residual mean +0.015, sd 0.062, z sd 0.92, so the model is calibrated where it matters. Residuals show no trend with donor mole fraction (corr 0.10), so a fraction-asymmetric excess term is not supported. Domain prior still used: literature pure viscosities as latent-component prior means, and a negative Grunberg-Nissan excess for H-bond-asymmetric pairs.
**Decision:** Continue latent_ei2 from its saved state, with the same script and no changes. No alternative model beat it in the LOO checks, and I see no open question whose answer would change the method.
**Hyperparameters:** Unchanged (state continue): sx=0.3, se_grid [.02..0.12], with sg and se chosen by marginal likelihood at each step; ei_noise=True; xi=0.
**Block length:** 40 measurements. That works through the remaining pairs with non-negligible PI, mainly the ether, pentane, methylal, acetone and MeCN pairs, and still leaves 126 to re-plan, e.g. a heavier-tailed predictive or deliberate sampling of untouched low-viscosity components if misses cluster.
**Review:** Any value below 0.434? Check the z-scores of the new points (state hist), especially for pairs with pre-measure sd below 0.03, where overconfidence would matter. Check whether the misses share a component type that would call for a component-specific excess term.

Proposal: `{"script": "work/latent_ei2.py", "label": "latent_ei2", "args": {"seed": 0, "se_grid": [0.02, 0.03, 0.05, 0.08, 0.12], "sx": 0.3, "ei_noise": true}, "n_points": 40, "state": "continue", "warm_start": false}`


Runner: Block finished: 40 paid measurements; 126 remain.

## Block 5 (after 124 of 250 measurements)

**Budget:** 124 used / 250 total; 126 remaining.
**Question:** The calibrated model says the pool's low end has been found. How should the remaining budget be spread to give the best chance that some unmeasured pair turns out below 0.434 cP?
**Mode:** exploit (endgame lottery over the lowest-predicted, distinct pairs).
**Gain:** Small. With the pair effect included, the summed probability of improvement over the pool is about 0.06; the best unmeasured predictions are 0.54–0.60 cP. This block maximises the chance of a hit and avoids spending points on pairs that cannot realistically win.
**Evidence:** Block 4 (40 EI points) did not beat 0.434 cP. Its best values were 0.441 (225, MeCN 0.73 + C3 alcohol) and 0.445 (4959). Pre-measure z had mean -0.09 and sd 0.98, so the model is still calibrated. Nine of the 40 points were EI tail picks with sd > 0.1 and predictions of 0.9–3.6 cP; none came near the best. On pairs involving heuristic (unknown) components the LOO z has sd 1.7 and is skewed upward (3.2, 4.1, 5.7), so the Gaussian downside tail that EI relies on is not supported there. Offline LOO on all 124 points tested a component-specific excess and a pair-level residual: marginal likelihood changed only slightly and LOO density in the low region did not change. ML picks se=0.01 with a pair effect sp=0.03 (nll -221.05 vs -219.06 without). A larger sg (0.5) makes EI chase extrapolated pairs with sd > 0.3, so the sg grid stays at ≤0.25. Domain prior unchanged: literature pure viscosities as latent prior means, and a Grunberg-Nissan excess.
**Decision:** Switch to work/latent_ei3.py, warm-started on all 124 points. Its predictions match latent_ei2 exactly when sp=0 (max difference 1e-14). Two changes. First, a pair-level random effect shared across all fractions of one component pair, so measured pairs' other fractions lose uncertainty and new pairs are preferred. Second, probability of improvement instead of EI as the acquisition, so it ranks low-mean pairs rather than high-sd tails. Its first picks are 159, 4978, 3492, 1404, 3448, 528 and 4138, predicted at 0.54–0.60 cP.
**Hyperparameters:** sg_grid [.1,.15,.25]; se_grid [.01,.02,.03,.05,.08,.12]; sp_grid [0,.03,.05], all chosen by ML at each step. sx=0.3, xi=0. The other settings are inherited from v2: sk .08, sk1 .15, su .35, ell 1.5, b0 .03, sb .1.
**Block length:** 40. That works through roughly the top 40 PI pairs, which make up most of the remaining PI mass, and leaves 86 points to re-plan. One option then is to concentrate on the neighbourhood of any new low value or of the components that keep producing the lowest residuals.
**Review:** Any value below 0.434? z of the new points (hist entries are (i, mean, sd_incl_se, logPI, (sg,se,sp))). Is sp still selected? Did PI keep to distinct pairs? If the misses are systematically positive at the low end (optimism), consider shifting to pairs of specific components with consistently negative residuals.

Proposal: `{"script": "work/latent_ei3.py", "label": "latent_pi3", "args": {"seed": 0, "sg_grid": [0.1, 0.15, 0.25], "se_grid": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12], "sp_grid": [0.0, 0.03, 0.05], "sx": 0.3, "acq": "pi", "xi": 0.0}, "n_points": 40, "state": "restart", "warm_start": true}`


Runner: Block finished: 40 paid measurements; 86 remain.

## Block 6 (after 164 of 250 measurements)

**Budget:** 164 used / 250 total; 86 remaining.
**Question:** The model has stayed calibrated and says no unmeasured pair is likely to beat 0.434 cP. How should the endgame lottery be ranked so points are not wasted on candidates whose downside the model overstates?
**Mode:** exploit (endgame lottery).
**Gain:** Small. Under v3 the summed probability of improvement over the pool is 0.003 to 0.02, depending on the misfit floor. The lowest unmeasured predictions are 0.55 to 0.62 cP. This block spends points on the lowest-mean, distinct pairs, where any heavier-than-Gaussian lower tail would show up. It avoids the 1 to 8 cP heuristic picks that v3's PI now ranks first.
**Evidence:** Block 5 (40 PI points) did not improve the best. Its best was 0.506 (1404). Pre-measure z had mean -0.10 and sd 0.88, so the model remains calibrated. I ran an exact GP leave-one-out on all 164 points (work/loo6.py, hyp .25/.01/.03):
- Known-component pairs (n=137): z mean -0.03, sd 0.92. 2.2% have z < -2, including -3.0 (225) and -2.6 (1116), a slightly heavy lower tail.
- Pairs with a heuristic (unknown) component (n=27): z mean +0.53, sd 1.81. None has z < -1.3, and 15% have z > 2. Their error is skewed upward, so the Gaussian downside that PI credits them with is not real.
- Other fractions of an already measured pair (n=16): z sd 1.07, so the pair effect is calibrated.

The extreme fractions of every top pair (1352, 225, 4959, 4072, 4109, 262, 4120) are already measured; their remaining fractions are predicted at 0.5 cP or higher. Domain prior is unchanged: literature pure viscosities as latent means, and a Grunberg-Nissan excess.
**Decision:** Switch to work/latent_pi4.py, warm-started on all 164 points. It uses the same v3 model for the mean and for the ML hyperparameters. Only the acquisition differs:
- The sd is recomputed with the heuristic components' prior sd set to su_acq=0.1, which removes their spurious downside.
- The point-level misfit floor is se_acq=0.03, which reflects the observed heavier lower tail.

Its first offline picks are 1535, 402, 892, 1337, 2816, 4874 and 4118, predicted at 0.55 to 0.61 cP, with summed PI about 0.018.
**Hyperparameters:**
- Grids are the same as block 5 (sg .1/.15/.25, se .01 to .12, sp 0/.03/.05) and are chosen by ML at each step; sx .3, xi 0.
- New settings: su_acq 0.1, based on the absence of downside misses among the 27 heuristic-pair points; se_acq 0.03, roughly the sd inflation that makes 2 of 137 points at z < -2.6 plausible.
- Offline, se_acq rather than su_acq drives the ranking: with se_acq=.01 the top picks become high-mean, extrapolated pairs.
**Block length:** 43, half the remaining budget. That is enough to work through the lowest-mean distinct pairs and see whether any lower-tail surprise appears near them, which would focus the last 43 points on that pair or component. It leaves the final block to be re-planned.
**Review:** Any value below 0.434? z of the new points (hist entries are (i, mean, sd_acq, logPI, hyp)). Are the misses on the low side clustered on certain components (e.g. 21, 22, 41, 59)? If so, take the last block around that component's partners.

Proposal: `{"script": "work/latent_pi4.py", "label": "latent_pi4", "args": {"seed": 0, "sg_grid": [0.1, 0.15, 0.25], "se_grid": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12], "sp_grid": [0.0, 0.03, 0.05], "sx": 0.3, "acq": "pi", "xi": 0.0, "su_acq": 0.1, "se_acq": 0.03}, "n_points": 43, "state": "restart", "warm_start": true}`


Runner: Block finished: 43 paid measurements; 43 remain.

## Block 7 (after 207 of 250 measurements)

**Budget:** 207 used / 250 total; 43 remaining.
**Question:** Is there any sign of a low-side surprise that should focus the last points on one component, or should the downside-aware PI lottery simply continue?
**Mode:** exploit (endgame lottery).
**Gain:** Small. The model is calibrated and the lowest unmeasured predictions are 0.57 to 0.65 cP, all other fractions of pairs already measured (sd about 0.01 in log10). Any gain needs a lower-tail surprise on a new pair. The downside-aware PI ranks those pairs best.
**Evidence:**
- Block 6 (43 points, latent_pi4) did not improve on 0.4344 cP (idx 1352). Its best value was 0.534 (618, comps 36+51, z -1.57).
- Pre-measure z: mean +0.28, sd 1.19. The misses are upward: 3414 at +5.6, 573 at +3.3, 1193 at +2.3. The low-side misses are mild (largest -1.6) and spread across components 36/51, 22/13, 41/51 and 7/41, with no cluster.
- So there is no component to concentrate on, and the model is, if anything, slightly optimistic. That does not change which candidates to rank.
- work/low6.py on all 207 points: 12 unmeasured candidates are predicted below 0.6 cP, all near-duplicates of measured pairs.
- Domain prior is unchanged: literature pure viscosities as latent means, and a Grunberg-Nissan excess.
**Decision:** Continue latent_pi4 from its saved state, with the same script and settings. No offline alternative beat it, and the review found no cluster that would justify a component-focused block.
**Hyperparameters:** Unchanged (state continue): su_acq 0.1, se_acq 0.03; sg/se/sp grids chosen by ML at each step; sx 0.3; xi 0.
**Block length:** 25. That leaves 18 for a final review, in case a low-side surprise appears and its pair or component neighbours deserve the last points.
**Review:** Any value below 0.434? Any z below -2, and on which components? If one appears, spend the final 18 on that pair's other fractions and that component's partners. Otherwise continue latent_pi4 to the end.

Proposal: `{"script": "work/latent_pi4.py", "label": "latent_pi4", "args": {"seed": 0, "sg_grid": [0.1, 0.15, 0.25], "se_grid": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12], "sp_grid": [0.0, 0.03, 0.05], "sx": 0.3, "acq": "pi", "xi": 0.0, "su_acq": 0.1, "se_acq": 0.03}, "n_points": 25, "state": "continue", "warm_start": false}`


Runner: Block finished: 25 paid measurements; 18 remain.

## Block 8 (after 232 of 250 measurements)

**Budget:** 232 used / 250 total; 18 remaining.
**Question:** Did block 7 turn up a low-side surprise that should focus the final points on one pair or component?
**Mode:** exploit (final endgame lottery).
**Gain:** Small. The model is calibrated and gives a low summed probability of beating 0.4344 cP. The downside-aware PI keeps choosing the distinct pairs most likely to beat it.
**Evidence:**
- Block 7 (25 points, latent_pi4 continue) did not improve the best. Its best value was 0.581 cP (2895, comps 1+87). The best overall is still 0.4344 cP (idx 1352).
- Pre-measure z (work/rev8.py): mean -0.10, sd 0.84, so the model is calibrated.
- No z fell below -2. The lowest were -1.54 (2208, comps 8+161), -1.29 (388, 26+94) and -1.02 (2152, 22+71). These are spread across different components, and their values (0.60 to 0.73 cP) are far above the best. There is no cluster to focus on.
- The only large miss was upward: 1315 at z +2.5.
- Domain prior is unchanged: literature pure viscosities as latent means, and a Grunberg-Nissan excess.
**Decision:** Continue latent_pi4 from its saved state with the same script and settings, as planned at the last review. Nothing contradicts it, and no offline alternative ranks better.
**Hyperparameters:** Unchanged (state continue): su_acq 0.1, se_acq 0.03; sg/se/sp grids chosen by ML at each step; sx 0.3; xi 0.
**Block length:** 18, the whole remaining budget. That is too few points for another review cycle to change the method.
**Review:** None remaining; the campaign ends with this block.

Proposal: `{"script": "work/latent_pi4.py", "label": "latent_pi4", "args": {"seed": 0, "sg_grid": [0.1, 0.15, 0.25], "se_grid": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12], "sp_grid": [0.0, 0.03, 0.05], "sx": 0.3, "acq": "pi", "xi": 0.0, "su_acq": 0.1, "se_acq": 0.03}, "n_points": 18, "state": "continue", "warm_start": false}`


Runner: Block finished: 18 paid measurements; 0 remain.
