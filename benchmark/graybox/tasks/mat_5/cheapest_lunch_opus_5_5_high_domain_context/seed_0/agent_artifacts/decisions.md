# Decisions

## Block 1 (after 0 of 100 measurements)

**Budget:** 0 used / 100 total; 100 remaining.
**Question:** How does CO2 uptake (0.15 bar, 298 K) vary across this MOF pool, and does the pore-size prior (small accessible pores give high uptake) hold? The answer decides which features a pool surrogate should focus on.
**Mode:** both (explore the pool's structure, and exploit the prior region).
**Gain:** a 15-point seed for a pool GP/BO. Half of the points are aimed at the region the prior expects to hold the high-uptake MOFs.
**Evidence:** No measurements. This is an indexed pool of 8540 MOFs with 327 descriptors (graph indices, Zeo++ pore geometry, RACs). Domain prior, not a measurement: low-pressure CO2 uptake should be highest for accessible small pores (Df about 3.3–5 Å, Di about 4.5–7 Å), with strong binding sites from metal or ligand chemistry (RAC chi/Z/I descriptors). Nonporous or inaccessible frameworks (ASA=0) should be near zero, and very large pores should be low. Assumption: the objective is deterministic (noise absent).
**Decision:** An experiment with fixed_points. (a) 8 medoids of k-means clusters, found in the top-20-PC space of signed-log standardized features (90% of variance), cover the whole pool. (b) 7 medoids from the prior region (Df 3.3–5, Di 4.5–7, ASA>0, POAVF>0.1, density<2.2; 1992 MOFs) cover that region; one prior medoid duplicated a diverse medoid and was dropped. If the prior-region points beat the diverse ones, the next block focuses the surrogate or search on the small-pore region. If they do not, the next block relies on a data-driven GP over the full pool.
**Hyperparameters:** k=8 per set, kmeans2 seeds 0 and 1, computed in work/design.py. Indices are listed in args.points.
**Block length:** 15 measurements: enough to fit a first surrogate and compare it with the prior, while leaving 85 for model-guided search.
**Review:** Rank the observed y against pore features (Df, Di, POAVF, density, ASA) and RAC chemistry. Fit a GP on reduced features (top PCs plus geometry) and check leave-one-out prediction. Then choose gp_bo on the pool (warm start) or a custom GP on a feature subset.

Proposal: `{"script": "lib/fixed_points.py", "label": "init", "args": {"seed": 0, "points": [6447, 8362, 2378, 6804, 6699, 6921, 2005, 5996, 3455, 7978, 7512, 5676, 6080, 4907, 4195]}, "n_points": 15, "state": "restart", "warm_start": false}`


Runner: Block finished: 15 paid measurements; 85 remain.

## Block 2 (after 15 of 100 measurements)

**Budget:** 15 used / 100 total; 85 remaining.
**Question:** Can a surrogate built from geometry and chemistry features find MOFs that beat the incumbent uptake of 7.16 mol/kg (idx 4907)?
**Mode:** exploit (model-guided), which also tests the surrogate.
**Gain:** The model ranks unused candidates by expected improvement, so this block should find more high-uptake MOFs than the 1-in-15 hit rate of the seed design. It also gives 20 more points for checking whether the model ranks well.
**Evidence:** Block 1 was 15 points with uptake from 0.09 to 7.16 mol/kg; median about 1.3. The prior region (Df 3.3–5, Di 4.5–7) produced the best point (4907: Df 4.07, Di 5.0, rho 1.31, metal Z=39, no functional groups) and 3 of the top 4. This weakly supports the pore-size prior, but some prior-region points were low (3455 at 0.09, 7978 at 1.05). Geometry alone does not predict uptake: the LOO Spearman of a GP on geometry features is −0.9 on log uptake. With n=15 the rank correlations are dominated by one point. Their direction is that lower metal electronegativity (mc-chi-0 ρ=−0.66) and linker RACs go with higher uptake, which fits the domain idea that strongly polar or charged sites drive low-pressure CO2 uptake. Offline LOO comparison of feature sets, all as Matern-5/2 GPs on log(uptake+0.05): PCs of all features gave ρ 0.5–0.6 and rmse 0.8. The domain subset gave ρ 0.60 and rmse 0.56 against sd 0.97. That subset is 9 Zeo++ geometry features (ASA, Df, Di, Dif, POAV, POAVF, density, grav SA, grav POV) plus depth-0 RACs for the metal (chi, Z, T, S), the linker (chi, Z, T, S, alpha) and the functional groups (chi, Z, T, S, alpha). The raw target was worse (ρ ≤ 0.58, rmse ≈ sd). These are 15-point estimates and very uncertain. In this sandbox, botorch's fit_gpytorch_mll crashes (threadpoolctl cannot read /proc/self/maps), so lib gp_bo might crash in the controller too. A 327-dim min-max ARD GP would also be poorly suited.
**Decision:** Switch to a custom optimizer, work/poolgp.py: a numpy/scipy GP (Matern-5/2, isotropic lengthscale, MAP fit with weak priors, refit before every proposal) on the 23-feature domain subset. Features are signed-log transformed and standardized over the pool. The target is log(uptake+0.05), and the optimizer takes the EI maximizer over all unused candidates. It is warm-started with all 15 points. The first proposal is idx 2977, a K-based MOF with Df 3.8 and Di 5.9, in line with the low-metal-electronegativity trend.
**Hyperparameters:** ard=false, because the isotropic and ARD LOO results were equal and isotropic is safer with about 20 points in 23 dims. offset=0.05 keeps the log finite near zero uptake. xi=0, because EI on a 1-σ log scale already explores. n_restarts=6 for the MAP fit. check.py passed with --warm 15.
**Block length:** 20 measurements, enough to see whether EI keeps finding uptake above about 3 mol/kg, while leaving 65 to change the features or kernel, or to switch to local exploitation around a family of good MOFs.
**Review:** Compute the hit rate (uptake >3 and >7.16) and the best value. Redo the LOO on 35 points: iso vs ARD, the subset vs adding depth-1 RACs, and whether geometry still hurts. Check whether the picks cluster around one chemistry (e.g. alkali or lanthanide metals). If the model's ranking holds, continue gp. If it does not, try ARD or neighbour-based exploitation around the best hits.

Proposal: `{"script": "work/poolgp.py", "label": "gp", "args": {"seed": 0, "ard": false, "offset": 0.05, "xi": 0.0, "n_restarts": 6}, "n_points": 20, "state": "restart", "warm_start": true}`


Runner: Block finished: 20 paid measurements; 65 remain.

## Block 3 (after 35 of 100 measurements)

**Budget:** 35 used / 100 total; 65 remaining.
**Question:** Is the GP-EI on the 23-feature domain subset still finding new high-uptake MOFs, or has it run out of the high-uptake families it has found?
**Mode:** exploit (continue the model-guided search that is working).
**Gain:** Block 2 found 9 of its 20 picks above 3 mol/kg and 5 above 6. It raised the best from 7.16 to 9.29 (idx 5725). If EI keeps that pace, another 20 picks should turn up a few more MOFs above 6 and could beat 9.29.
**Evidence:** The high hits share chemistry: Y-based MOFs (metal Z=39, x88=1521) have 6 measured with mean 6.9 and max 9.29 (5725, 3429, 3430, 4907, 3431, 1480). Metals with Z^2 = 400 (Ca), 1444 (Sr) and 4225 (Tb) also gave 5.5–6.2. All of the best have Df 4–6, Di 5–7, no functional groups and density 1.0–1.3. This fits the domain prior on pore size, and the electropositive-metal trend from block 1 holds up. 38 unused Y MOFs remain, many of them nonporous (ASA=0). I redid the leave-one-out test on the 35 points, using log(uptake+0.05) as the target (sd 0.99). The 23-feature subset with an isotropic kernel gave ρ=0.53 and rmse 0.87. ARD on the same subset gave ρ=0.43 and rmse 0.91. The top 20 principal components gave ρ=0.52 and rmse 0.91, and geometry plus metal features gave ρ=0.35. The isotropic subset model is still the best or tied, so changing the kernel or features is not justified. Its moderate LOO accuracy is offset by an EI search that has worked well in practice.
**Decision:** Continue the `gp` label (work/poolgp.py, saved state) unchanged.
**Hyperparameters:** Unchanged: iso Matern-5/2 with MAP refit at every step, target log(up+0.05), xi=0, n_restarts=6. These were chosen in block 2 and the LOO check still supports them.
**Block length:** 20 measurements. That gives enough picks to measure the hit rate again, and leaves 45 to switch to local exploitation, such as nearest neighbours in feature space of the top Y and Ca/Sr/Tb MOFs, if EI stalls.
**Review:** Check the hit rate above 6 and whether the best has improved. Check whether the picks have used up the Y family or moved into other metals. If the hit rate drops sharply, consider a neighbour or trust-region exploitation around 5725 and 3429, or a model with a metal-identity feature.

Proposal: `{"script": "work/poolgp.py", "label": "gp", "args": {"seed": 0, "ard": false, "offset": 0.05, "xi": 0.0, "n_restarts": 6}, "n_points": 20, "state": "continue", "warm_start": false}`


Runner: Block finished: 20 paid measurements; 45 remain.

## Block 4 (after 55 of 100 measurements)

**Budget:** 55 used / 100 total; 45 remaining.
**Question:** Is the yttrium (Y) family reliably high-uptake? If it is, can untested porous Y MOFs beat the best of 9.29? Block 3's EI stopped finding high values (0 of 20 above 6, best 5.61).
**Mode:** exploit (with a built-in test of the hypothesis that metal identity drives uptake).
**Gain:** Y MOFs measured so far gave 9.29, 8.07, 7.16, 7.16, 6.89, 5.61 and 2.85 (mean 6.7, n=7). No other metal averages above 4.8. If about 1 in 8 new Y MOFs beats 9.29, then 10 picks have roughly a 50–70% chance of a new best, and most picks should land above 6. Block 3 managed none above 6.
**Evidence:** Block 3 (iso GP-EI, 20 points) explored other metals: mean 2.6, max 5.61, no improvement. I compared GPs on the 55 points by leave-one-out (LOO), with the target log(uptake+0.05). The current isotropic GP gave ρ=0.48 and rmse 0.90. Isotropic Matern with metal-match kernel (same-metal similarity): ρ 0.45; top-10 picks scored mean 5.70. A 3-group ARD GP (geometry / metal / linker+functional groups) did better: ρ 0.56, rmse 0.84. The 10 points it ranked highest out of sample had a mean uptake of 5.48, versus 4.68 for the isotropic GP. Its fitted lengthscales were 38 for geometry, which makes geometry nearly irrelevant among the porous MOFs picked so far, 0.84 for metal (dominant) and 2.2 for linker and functional groups. The group-ARD model predicts about 7.5 for every unused Y MOF without functional groups. Its EI is near zero because the mean is below 9.29 and the sd is overconfident. That is why a model-driven EI will not test these MOFs, and it is better to pick them directly. Domain prior: a pore-limiting diameter of about 4–6 Å, a high void fraction and electropositive metals favour low-pressure CO2 uptake. Caveat: 1480 has Y and geometry close to 5725's yet gave only 2.85, so this family is not uniformly good.
**Decision:** Experiment and exploit. Switch to lib/fixed_points with 10 unused porous Y MOFs:
- 3432, 3433 and 3434, siblings of the 3429–3431 series that scored 8.07, 7.16 and 6.89;
- 1278, close in geometry to 4907 (7.16);
- 3984, 4699 and 5141, with Df about 4.5 and Di 5.4–5.9 like 5725, and with a small or halogen/O functional group;
- 4134 and 3441, high-ASA Y MOFs;
- 4801, an ultramicroporous Y MOF with ASA=0 and rho 2.0, a cheap test of whether nonporous-by-N2 Y MOFs can be high.

If most picks come in above 6, the metal-dominant group-ARD model is confirmed: use it, or the remaining Y and Sr/Ca/Tb, for the rest of the budget. If they average about 4 or less, family membership is not enough, and I return to a model-guided search with geometry kept informative.
**Hyperparameters:** Not applicable (fixed list). Seed 0.
**Block length:** 10 measurements. That is enough to estimate the Y-family hit rate and maybe beat 9.29, while leaving 35 to exploit the answer with a group-ARD GP.
**Review:** Count Y picks above 6 and above 9.29. Refit the group-ARD GP with the new data. Write a custom group-ARD poolgp, e.g. a variant with geometry lengthscale capped or UCB, run check.py on it, and use it for most of the remaining 35 points.

Proposal: `{"script": "lib/fixed_points.py", "label": "yfam", "args": {"seed": 0, "points": [3434, 3433, 3432, 1278, 3984, 4699, 5141, 4134, 3441, 4801]}, "n_points": 10, "state": "restart", "warm_start": false}`


Runner: Block finished: 10 paid measurements; 35 remain.

## Block 5 (after 65 of 100 measurements)

**Budget:** 65 used / 100 total; 35 remaining.
**Question:** Is the metal coordination environment the factor that separates high- from low-uptake MOFs, beyond metal identity? Specifically: the depth-2 metal RACs x86 (mc-chi-2), x94 (mc-I-2) and x98 (mc-T-2). If so, can a GP on a compact metal/coordination/porosity feature set find MOFs above 9.29 outside the Y family, which is nearly used up?
**Mode:** both (model-guided exploitation with a new feature set, which tests the coordination hypothesis).
**Gain:** Under this model, the top EI candidates have roughly a 10–20% chance each of beating 9.29. That is probably optimistic. The picks will show whether low-coordination, electropositive-metal MOFs outside Y (Sc, Ca, Sr, mixed metals) give uptake above 6. If they do, keep this model for the final ~23 points. If they do not, go back to direct exploitation of the remaining Y and Y-like candidates (7353, 6449, 4134-series and 5725-series analogs).
**Evidence:** Block 4 (10 Y MOFs) gave 4134 at 9.20, nearly tying the best; 5141 gave 6.33 and the rest 0.3–4.8 (mean 4.1). So Y alone is not enough. Within the 17 Y MOFs, uptake rank-correlates strongly with x98 (ρ=−0.85), x94 (−0.82) and x86 (−0.82). Every non-functionalized Y MOF with x94≤5 scored 6.9–9.3, while those with x94≥7 scored 0.3–4.4. Across all metals the top 12 MOFs nearly all have x94≤5 and metal chi²≤1.5. A plausible physical reading is that low-coordinated, electropositive metal sites drive strong CO2 binding at 0.15 bar. This fits the domain prior that polar or charged sites dominate low-pressure uptake. I compared LOO GP feature sets on 65 points (target log(up+0.05), isotropic Matern-5/2, Spearman ρ and rmse; sd 0.99):

| Feature set | ρ | rmse | Mean uptake of top-10 OOS picks |
|---|---|---|---|
| Old 23-feature default | 0.40 | 0.93 | 5.01 |
| Old default + mc2 | 0.45 | 0.90 | 5.83 |
| New 10 features | 0.64 | 0.76 | 6.11 |
| New 10, ARD kernel | 0.54 | 0.86 | — |

The new set is ASA x69, Df x71, density x79, metal chi² x84, Z² x88, x86, x94, x98, and functional-group chi x212 and Z x216. This feature set was partly selected on these same 65 points, so the gain is somewhat optimistic. Posterior means top out near 5.8, so 9.29 is an outlier for every model. The heavier lanthanide analogs of 5725 (Tb 5721 gave 6.17 vs Y 9.29) suggest a mol/kg mass penalty, so Ln analogs are unlikely to win. The first EI pick is idx 1781: Sc (lighter than Y), x94=4, Df 3.85, with an N-containing functional group.
**Decision:** Switch: restart work/poolgp.py (unchanged code, already check-passed) under the new label gp2 with the 10-feature set. Warm-start it with all 65 points. If the picks reach above 6, the coordination hypothesis holds. If they stay near the block-3 level (mean ~2.6), the model is overfit to Y, and I will hand-pick the remaining Y/Y-like MOFs.
**Hyperparameters:** features as above; ard=false (isotropic LOO beat ARD, 0.64 vs 0.54); offset 0.05, xi 0 and n_restarts 6, as in the gp block.
**Block length:** 12 measurements. That is enough to judge the new model's hit rate, and leaves 23 for final exploitation.
**Review:** Look at the hit rate above 6 and above 9.29, and which metals and x94 values the picks used. Rerun work/loo6.py and work/post6.py with 77 points. Decide between continuing gp2 and a fixed list (7353, 6449, 4629, 4136, 4240, 7574, 4765, ...) for the final blocks.

Proposal: `{"script": "work/poolgp.py", "label": "gp2", "args": {"seed": 0, "features": ["x69", "x71", "x79", "x84", "x88", "x86", "x94", "x98", "x212", "x216"], "ard": false, "offset": 0.05, "xi": 0.0, "n_restarts": 6}, "n_points": 12, "state": "restart", "warm_start": true}`


Runner: Block finished: 12 paid measurements; 23 remain.

## Block 6 (after 77 of 100 measurements)

**Budget:** 77 used / 100 total; 23 remaining.
**Question:** The gp2 model failed. Is there another light, electropositive-metal family with the "Y-like" coordination signature (x94 ≤ 5.3, x98 about 70–100, porous, no functional group) that can reach 9.29? The candidate is Mg (x88=144): 124 candidates, none measured yet.
**Mode:** both. The block tests the Mg hypothesis, and each pick is a plausible new best.
**Gain:** If any Mg pick scores above 7, the remaining ~17 go to the other unmeasured Mg siblings (3212, 3563, 4161, 2970, 7319, 4091/4092, 1193). If all score below 5, Mg is ruled out cheaply and I finish with the remaining Y/Y-like and best-of-family picks. Y 7353 is the last porous Y MOF with low x98 (105). It has a functional group, but by the Y trend it is a plausible new best.
**Evidence:** Block 5 (gp2, 12 EI picks) gave at most 5.02 and mean about 2.5, with no improvement. The 10-feature GP does not generalize beyond Y. Across all 77 points, every value ≥6 comes from a metal with chi² ≤ 1.5 (Y, Sr, Ca, Tb), x98 72–96, x94 4–6, ASA > 1100 and no functional group. Within Y, x98 ≤ 96 gave 6.9–9.3 (7 MOFs) and x98 ≥ 105 gave ≤ 5.6 (10 MOFs). Ln analogs of the Y winners scored much lower: Tb 5721 6.17 vs 9.29, La 4136 4.07 vs 9.20. That fits a mol/kg mass penalty, so I avoid Ln analogs. Sr/Ca analogs plateau at 5–6. Mg is lighter and electropositive (chi² 1.72). Its 15 porous, non-functionalized members have x94 5.33 and x98 86–92, matching the winning signature, but the family has never been tested. The domain prior also favours electropositive or open metal sites for low-pressure CO2 uptake. Counter-evidence: Sc (chi² 1.85) 1781 gave 1.16, though it had poor geometry and a functional group.
**Decision:** Experiment and exploit with lib/fixed_points. Five non-functionalized Mg MOFs with different geometry:
- 3210: ASA 2214, Df 4.66, rho 0.82; the closest to 5725
- 7318: ASA 2077, Df 4.93, rho 0.93
- 5058: ASA 1643, Df 4.65
- 5920: Df 5.47, x98 86
- 3562: smaller pore, Df 3.88, ASA 934, to test pore size

Plus Y 7353.
**Hyperparameters:** Not applicable (fixed list, seed 0).
**Block length:** 6 measurements. That is enough to accept or reject Mg and still leaves 17 to exploit the answer.
**Review:** Look at the Mg values and the Df/ASA trend within them. If any Mg is ≥7, pick the closest unmeasured Mg siblings. Otherwise, finish with Ca 4629, Y 6449/3000, Ln 4240/7574 (x94 4, ASA 1860–2290) and a few top Sr/Ca analogs.

Proposal: `{"script": "lib/fixed_points.py", "label": "mgprobe", "args": {"seed": 0, "points": [3210, 7318, 5058, 5920, 3562, 7353]}, "n_points": 6, "state": "restart", "warm_start": false}`


Runner: Block finished: 6 paid measurements; 17 remain.

## Block 7 (after 83 of 100 measurements)

**Budget:** 83 used / 100 total; 17 remaining.
**Question:** Mg is now confirmed as the second-best family. Which of its coordination variants, or which remaining light-metal family, has any real chance of beating 9.29?
**Mode:** both, but mainly exploit. Every pick is a plausible high scorer, and each one also tests a single hypothesis.
**Gain:** A new best is unlikely to come from any single pick (I estimate P(>9.29) at no more than about 5% each). This block spreads that chance over 7 different hypotheses. The results decide the final 10: if a variant scores ≥ 8, its siblings get the remaining budget; otherwise I take the best-mean Mg/Y siblings.
**Evidence:** Block 6 Mg results: 7318 7.64, 5058 6.32, 3562 6.07, 3210 5.40, 5920 3.63 (mean 5.8, the highest family mean after Y). The low-density, large-pore member (5920, rho 0.74) was worst. The Y MOF with a functional group (7353) gave 3.79, so functional groups hurt again. The unused Y members with the signature all have x98 = 108, and that group previously scored ≤ 5.6. Within the 36 porous low-chi points, Spearman ρ with uptake is −0.50 for x94, so lower coordination is better. A subset GP (LOO ρ 0.58, rmse 1.85 vs sd 2.19) ranks Ln analogs of the 3429/3430 family first (mean about 6.7). It ignores the observed Ln mass penalty (Tb 5721 6.17 vs Y 9.29; La 4136 4.07 vs Y 9.20), so I include only one Ln pick, to test the GP. The domain prior favours light, electropositive metals, which supports Mg and the Li test.
**Decision:** Experiment and exploit with a fixed list:
- 7319: twin of 7318 (ASA 2063, Df 4.91); highest expected value, about 7.6.
- 4092: Mg with the 4134-type coordination (x94 6, x98 90), rho 1.12.
- 3212: Mg, intermediate pore (Df 4.42, rho 0.93), between 7318 and 3562.
- 8105: Mg with x94 4 / x98 48, larger pore (Df 7.9).
- 4013: Li (lightest metal), x94 5.5, x98 67, ASA 1157, no functional group. The only Li measured so far (7512, ASA 664) gave 2.33.
- 1193: Mg with a geometry like 7318 (ASA 1891, Df 4.25, rho 0.92), x98 80 and a small functional group (x216 39).
- 3435: Ln analog of the Y 3430/3431 family (x94 4, x98 84), the subset GP's top pick.
**Hyperparameters:** Not applicable (fixed list).
**Block length:** 7 measurements. That leaves 10 to exploit whichever variant wins.
**Review:** If any variant scores ≥ 8, spend the rest on its siblings (4091 for 4092; 3563/4161 for 3212; 2970 for the 7318 group). If 3435 scores ≥ 7, try the other Ln x98=84 analogs (3436, 5880–5882). If Li 4013 scores ≥ 6, try 7510/7515/5290. Otherwise finish with the best-mean Mg siblings plus 3563, 4161 and Ca 4629.

Proposal: `{"script": "lib/fixed_points.py", "label": "mgfollow", "args": {"seed": 0, "points": [7319, 4092, 3212, 8105, 4013, 1193, 3435]}, "n_points": 7, "state": "restart", "warm_start": false}`


Runner: Block finished: 7 paid measurements; 10 remain.

## Block 8 (after 90 of 100 measurements)

**Budget:** 90 used / 100 total; 10 remaining.
**Question:** With 10 measurements left, which remaining family could still beat 9.29? Candidates: Ln analogs of the Y winners using the lanthanides closest to Y in size (Dy/Ho), the dense Mg x98=92 members, and Ca.
**Mode:** mostly exploit. Each pick is a plausible high value, and the first three also test one hypothesis (Ln analogs with a radius like Y's).
**Gain:** A new best is unlikely (each pick has a P(>9.29) of a few %). The results decide the last 4. If Dy 3090 or Ho 5722 scores ≥ 8, the rest go to their siblings (Dy 7919, Er 7918, Tm 4133; Er 5723, Tm 71). Otherwise the rest go to the best-scoring Mg, Ca or Sr siblings.
**Evidence:** Block 7: Mg 7319 7.68 (twin of 7318 7.64), Mg 8105 7.48 (x94 4, x98 48, Df 7.9; the only Mg member of that type without a functional group), Mg 3212 6.90 (dense, small pore), Mg 4092 3.99, Li 4013 4.14, Mg+FG 1193 3.93, Sm 3435 5.14. Within Mg x94 5.33 / x98 92, dense small-pore members do well: 3212 6.90, 3562 6.07, 7318/7319 7.6. Low-density, large-pore members do badly: 5920 3.63, 3210 5.40. Functional groups hurt everywhere: 10 N-FG points average 2.51, and 57 points without a FG average 4.01, with every value ≥ 6.9 among them. Ln analogs score lower than Y: Tb 5721 6.17 vs 9.29, La 4136 4.07 vs 9.20, Sm 3435 5.14 vs 7.16. La's drop is larger than its mass alone explains, which suggests a size effect. That is why I test Dy (chi² 1.49, identical to Y) and Ho. The domain prior (light, electropositive metal sites) supports Mg and Ca.
**Decision:** Fixed list of 6 (lib/fixed_points):
- Dy 3090: analog of 4134 (x94 6, x98 90, two-pore x72 8.96, ASA 1965).
- Ho 5722: direct analog of 5725 (ASA 1776, Df 4.62).
- Dy 4241: 5725-type with the 4134-like secondary pore (x72 7.46, ASA 2284).
- Mg 3563 and 4161: dense x98=92 members, between 3562 (6.07) and 3212 (6.90).
- Ca 4765: twin of 4764 (5.81) with larger ASA.
**Hyperparameters:** not applicable (fixed list, seed 0).
**Block length:** 6 measurements, leaving 4 to follow up on any hit before the budget ends.
**Review:** If any Ln pick scores ≥ 7.5, spend the last 4 on the same-type siblings with Dy/Ho/Er/Tm (7919, 7918, 4133, 4242, 4239, 5723, 71). Otherwise, finish with Ca 4629, Sr analogs, Er 7918, or Mg 2970 depending on results.

Proposal: `{"script": "lib/fixed_points.py", "label": "final1", "args": {"seed": 0, "points": [3090, 5722, 4241, 3563, 4161, 4765]}, "n_points": 6, "state": "restart", "warm_start": false}`


Runner: Block finished: 6 paid measurements; 4 remain.

## Block 9 (after 96 of 100 measurements)

**Budget:** 96 used / 100 total; 4 remaining.
**Question:** None of the open questions can still change the method. The only thing left is which 4 untested candidates have the best chance of beating 9.29.
**Mode:** exploit (last block).
**Gain:** Probably no improvement; each pick has a small chance (a few %) of beating 9.29. The picks cover the families with the highest means and one high-variance light-metal shot.
**Evidence:** In Block 8 every Ln analog scored below Y: Dy 3090 1.70, Ho 5722 5.35, Dy 4241 5.51. This confirms Ln << Y, so I drop the remaining Er/Tm siblings. Mg 3563 6.14 and 4161 6.30 are consistent with the dense Mg x98=92 family (6.1–7.7). Ca 4765 4.42 was below its twin 4764 (5.81). Per-metal means: Mg 5.96 (n=12), Y 5.11 (n=18), Ca 4.23, Sr 4.05; transition metals are all below 2.5. Functional groups hurt everywhere. Domain prior: light, electropositive metal sites favour uptake in mol/kg.
**Decision:** Fixed list (lib/fixed_points):
- Mg 2970: the last unmeasured Mg x98=92 member without a functional group (ASA 1709, Df 6.0).
- Ca 4629: the 6/4/90 type, the same as Ca 5860 (5.53) and Sr 4314 (6.23).
- Mg 5919: an untested Mg coordination (x94 6, x98 112), ASA 1727.
- Li 4633: the lightest metal, low coordination (2/2/12), ASA 2461; a high-variance shot.
**Hyperparameters:** not applicable.
**Block length:** 4, the whole remaining budget.
**Review:** The campaign ends. The best point is 5725 (9.29) unless a pick beats it.

Proposal: `{"script": "lib/fixed_points.py", "label": "final2", "args": {"seed": 0, "points": [2970, 4629, 5919, 4633]}, "n_points": 4, "state": "restart", "warm_start": false}`


Runner: Block finished: 4 paid measurements; 0 remain.
