# Decisions

## Block 1 (after 0 of 100 measurements)

**Budget:** 0 used / 100 total; 100 remaining.
**Question:** Where in the 8540-MOF pool does low-pressure CO2 uptake (mol/kg) concentrate, and does the domain prior (small/medium pores, moderate void fraction) hold? Which features does a surrogate need?
**Mode:** both (prior-weighted space-filling design).
**Gain:** a 20-point training set for a surrogate-guided pool search (custom GP/ensemble on reduced features) for the remaining 80, plus a likely-good incumbent from the prior region; tests whether non-porous (ASA=0, 1/3 of pool) and large-pore MOFs can be excluded.
**Evidence:** No measurements yet. Pool: 8540 candidates x 327 descriptors (graph indices, Zeo++ geometry, RACs). Domain context: y = -CO2 uptake at 298 K/0.15 bar. Prior (hypothesis, not measurement): at 0.15 bar, uptake is driven by strong binding sites/small pores (Df ~4-7 Å) and polar chemistry (RAC electronegativity differences), not by large surface area; non-porous (ASA=0, 33% of pool) likely near-zero; very large pores weak. Feature analysis: signed-log + standardize + PCA, 20 PCs explain 90% of variance.
**Decision:** experiment/initial design via fixed_points: 14 k-means medoids (in PCA-20 space) from the prior region (ASA>0, 3.8<=Df<=7.5, 0.2<=POAVF<=0.65; 3716 candidates), 4 medoids from other porous MOFs (large/very small pores), 2 from non-porous. Outcomes: if prior region dominates the top, focus the model-guided search there; if large-pore or non-porous points score well, widen the search region.
**Hyperparameters:** not applicable (chosen indices from work/design.py, seeds 0/1/2 for k-means).
**Block length:** 20 measurements — enough for a first surrogate with a few key features while leaving 80 for model-guided exploitation.
**Review:** rank y vs Df, Di, POAVF, density, ASA and RAC chi/D_func features; fit GP/RF on reduced features with LOO; build a custom pool BO (EI/TS over unused indices, possibly restricted to prior subset).

Proposal: `{"script": "lib/fixed_points.py", "label": "init_design", "args": {"seed": 0, "points": [2377, 1063, 3000, 275, 6080, 1602, 6272, 3730, 7358, 239, 6441, 6445, 7668, 6977, 3081, 1940, 3481, 7346, 927, 233]}, "n_points": 20, "state": "restart", "warm_start": false}`


Runner: Block finished: 20 paid measurements; 80 remain.

## Block 2 (after 20 of 100 measurements)

**Budget:** 20 used / 100 total; 80 remaining.
**Question:** Can a surrogate on reduced features find MOFs beating the incumbent 7.41 mol/kg (cand 6272), and which model family is predictive here?
**Mode:** exploit (model-guided), with built-in exploration via tree-spread UCB.
**Gain:** expected to find several MOFs in the 3-7+ mol/kg range and possibly beat 7.41; 15 new labels also test whether the RF surrogate's LOO rank skill (~0.65) holds on fresh proposals.
**Evidence:** Block 1 results: best 7.41 (6272: Df 5.3, Di 6.9, POAVF 0.39, density 1.43), then 5.02 (1063), 4.59 (6445), 3.90 (3000). Large-pore / high-POAVF MOFs all <1.3; non-porous (ASA=0 to probe) 233 and 927 gave 2.27 and 1.68, so the domain prior "non-porous ~ zero" is contradicted and they stay in the pool. Spearman with uptake: total gravimetric pore volume -0.51, Di -0.48, density +0.49, metal-centred RAC electronegativity D_mc_CRY-chi-3 -0.62: consistent with the domain prior (small pores, dense, strong binding sites). LOO on 20 points: GPs (ARD or isotropic, PCA/geo/all features) collapse to the mean (negative LOO Spearman); RF/ExtraTrees on PCA10 (signed-log, standardized descriptors) give Spearman 0.64-0.69; geometry-only trees ~0.1-0.2. Library gp_bo on all 327 raw dims would therefore be uninformative.
**Decision:** switch to a custom RF-UCB pool optimizer (work/rfbo.py), warm-started with all 20 measurements: RF on 10 PCs + 9 Zeo++ geometry descriptors, target = uptake, acquisition mean + 1.0*std over trees, refitted after every measurement over all unused candidates.
**Hyperparameters:** kappa 1.0 (dry run: 0.5 chased very dense ASA=0 extrapolations, 2.0 chased large-pore MOFs that the data already rule out; 1.0 picks Df 4-6.5 Å, mid-density MOFs); 500 trees, min_samples_leaf 2, max_features 0.5 (best LOO settings tested); n_pca 10 (best LOO). Seed 0.
**Block length:** 15 measurements: enough refits to see whether the model's picks beat block-1 medoids (median ~1.6) before the next review, leaving 65 to adjust kappa or the features.
**Review:** hit rate of the block (fraction >3 mol/kg) vs block 1; improvement over 7.41; refit LOO including new points; whether the picks cluster (narrow kappa) or wander into dense ASA=0 frameworks (then add a density cap or lower kappa).

Proposal: `{"script": "work/rfbo.py", "label": "rfucb", "args": {"seed": 0, "n_pca": 10, "kappa": 1.0, "n_trees": 500, "min_leaf": 2, "max_features": 0.5, "geo_idx": [56, 57, 58, 61, 64, 54, 60, 55, 59]}, "n_points": 15, "state": "restart", "warm_start": true}`


Runner: Block finished: 15 paid measurements; 65 remain.

## Block 3 (after 35 of 100 measurements)

**Budget:** 35 used / 100 total; 65 remaining.
**Question:** Can a better-specified surrogate (geometry + chemistry features chosen from the data) beat 7.41 mol/kg, or at least keep finding 4-6 mol/kg MOFs more reliably than RF-on-PCA did?
**Mode:** exploit, with UCB exploration.
**Gain:** more hits per measurement than block 2 (6 of 15 above 3 mol/kg); a chance to beat 7.41 by testing the high-density, low-metal-chi members of the linker family the model ranks highest, plus the few extreme D_lc-alpha MOFs similar to 8445.
**Evidence:** Block 2 (RF-UCB on PCA10 + geometry) found 6 of 15 above 3 mol/kg (block 1: 4 of 20), including 8445 = 5.96, but nothing beat 7.41. Refitting LOO on all 35 points, the old feature set dropped to Spearman 0.38 (RF geo+PCA10). PCA features now add little. ExtraTrees on 13 Zeo++ geometry descriptors + the 10 descriptors with the highest |Spearman|, re-selected inside each fold, reaches 0.57. Its top-8 LOO picks average 4.2 mol/kg, against 3.6 for the other models. Isotropic GP on geo5 + top-5 reaches 0.61, so the two are similar and I chose ET for its robustness. Strongest correlates are linker/functional-group RACs: D_lc-alpha-3 (x211) -0.61, sum-D_func-chi-3 -0.58, metal mc_CRY-chi-0 (x84) -0.58. Every hit has Df 3.9-5.3, POAVF 0.34-0.52, NASA 0 and x211 <= -12. One linker family (x211=-20.1, x239=-2.68; 50 pool members) gives 4.0-4.9 consistently unless its pores are large (5823: Df 6.4, 1.02). These results fit the domain prior: small pores and polar or functionalised linkers favour low-pressure uptake. Dry run: the new model picks unmeasured members of that family with higher density (1.3-1.7) and lower metal chi (x84 1.2-1.5), plus a dense outlier (400).
**Decision:** switch to a custom ExtraTrees-UCB pool optimizer (work/etbo.py; check.py OK), warm-started with all 35 points. The feature set is re-selected at every refit, and the target is log1p(uptake). Abandon label rfucb.
**Hyperparameters:** kappa 0.75: in the dry run, 0 sticks to near-duplicates and 1.0 already reaches outliers. k_sel 10 gave the best LOO (20/40 no better). 500 trees, min_samples_leaf 2, max_features 0.5 are the LOO settings. Seed 0.
**Block length:** 15 measurements. Each point's feedback should quickly prune the near-duplicate family members if they disappoint, and 50 points remain for a final, more exploitative phase.
**Review:** hit rate and best vs 7.41; whether picks stayed inside one family (if so, add diversity or raise kappa); LOO of etbo vs GP with 50 points; consider looking near 6272/8445 chemistry explicitly.

Proposal: `{"script": "work/etbo.py", "label": "etucb", "args": {"seed": 0, "k_sel": 10, "n_pca": 0, "kappa": 0.75, "n_trees": 500, "min_leaf": 2, "max_features": 0.5, "geo_idx": [56, 57, 58, 61, 64, 54, 60, 55, 59, 65, 66, 67, 68]}, "n_points": 15, "state": "restart", "warm_start": true}`


Runner: Block finished: 15 paid measurements; 50 remain.

## Block 4 (after 50 of 100 measurements)

**Budget:** 50 used / 100 total; 50 remaining.
**Question:** The x211=-20.1 linker family seems to top out around 4-5.5 mol/kg. Can the model find other chemistries that beat 7.41 mol/kg (cand 6272)?
**Mode:** exploit with mild exploration, moving away from the mined family toward new small-pore chemistries.
**Gain:** a fair chance of beating 7.41. The GP's top picks have predicted uptake around 4.1-4.5 with latent sd ~0.35 in log1p units, so each pick is roughly 1.3 sd below 7.41. Their chemistries are new (x211 -25.7, -24, -15.6, -12.9, -11.3), so each result also tells us whether the "negative linker/func RAC + small pore" rule extends beyond one family.
**Evidence:** Block 3 (ET-UCB) gave 11 of 15 picks above 3 mol/kg, the best hit rate so far, but its best was 6.15 (7135, x211=-76) and 12 picks came from the x211=-20.1 family (now 18 of 67 measured; values 0.8-5.5, median ~4.3). The incumbent 6272 (7.41) is chemically isolated: its nearest neighbours in RAC space are all unmeasured and far away, and every LOO model ranks it only ~27th-34th of 50, so the surrogates cannot explain it. LOO at 50 points: isotropic GP on geo5 (Df, Di, POAVF, density, POAV) + top-5 Spearman features has Spearman 0.655 and its top-10 average 4.34 (best). ET on geo13+10 has 0.60/4.24 and geo13+20 0.59/4.50. The selected features are x211, x239, x355, x171, x187 (linker/func RAC alpha/chi), with the strongest |rho| 0.67. This fits the domain prior that polar or functionalised linkers and small pores drive low-pressure uptake. With kappa >= 1, both models chased 40 candidates with Df=Di=0 but POAVF ~0.57 (missing Zeo++ geometry). These are extrapolations, so they are excluded.
**Decision:** switch to a custom GP-UCB pool optimizer (work/gpbo.py, check.py OK), warm-started with all 50 points. It is an sklearn GP with Matern 2.5 + white noise on log1p(uptake). The features are geo5 plus the top-5 by Spearman, re-selected at every refit. Candidates with Df<=0 are masked. Label etucb is retired, not crashed.
**Hyperparameters:** kappa 0.5: in the dry run kappa 0-1 give nearly the same diverse top picks (5491, 5025, 1544, 2091, 2090, 3629), and higher kappa drifts into the masked or extrapolated corners. k_sel 5 is the best LOO setting (10 gives 0.58, 20 gives 0.55). The GP hyperparameters are refitted by ML at every step. The posterior sd is latent (noise excluded). Seed 0.
**Block length:** 15 measurements, so per-point feedback can prune near-duplicates (e.g. 2090/2091) and test 4-6 new chemistries. That leaves 35 for a final phase chosen from the outcome.
**Review:** count of hits above 3 and above 5, and any value above 7.41. Check whether GP picks beat the ET block's hit rate. If new chemistries fail (<3), go back to ET/family exploitation near 7135/2778/2894. If one gives a new high, exploit its neighbourhood. Refresh the LOO comparison of GP and ET.

Proposal: `{"script": "work/gpbo.py", "label": "gpucb", "args": {"seed": 0, "k_sel": 5, "kappa": 0.5, "geo_idx": [56, 57, 61, 64, 60], "mask_col": 56}, "n_points": 15, "state": "restart", "warm_start": true}`


Runner: Block finished: 15 paid measurements; 35 remain.

## Block 5 (after 65 of 100 measurements)

**Budget:** 65 used / 100 total; 35 remaining.
**Question:** Has GP-UCB already used up the low-metal-chi (x84 = 1.72/1.49), x211 = -12, small-pore family it found, or are there unmeasured members (or neighbouring chemistries) above 8.66?
**Mode:** exploit. The UCB term adds mild exploration.
**Gain:** a good chance of beating 8.66. Block 4 found 6 of 15 picks above 7.4, and the family still has many unmeasured members with the same descriptors: 4355, 4356, 3656 and 2806 share the x86 = 13.36 / x150 = 48 linker, where all 4 measured MOFs gave 8.1-8.7, and 3210, 7318 and 5058 share the 15.65 linker of 7319 (7.68) and 3212 (6.90).
**Evidence:** Block 4 (GP-UCB) raised the best from 7.41 to 8.66 (cand 4230). It also found 6505 = 8.46, 8469 = 8.23, 6504 = 8.13, 3429 = 8.07 and 7319 = 7.68. In the data so far, all of these share metal RAC x84 = 1.72 or 1.49 (8 measured at 1.72: mean 7.2), x211 of about -12 (-10.4 for 4230), Df 4.1-4.9, POAVF 0.45-0.53 and density 0.86-1.02. The same metal with the x211 = -20.1 linker gave only about 4.9 (7531, 5027). Other low-chi metals did poorly (1.44: 2.3; 1.21: 1.7; 1.25: 3.8), so the metal effect looks specific to these two values. The earlier rule "x239 <= 0" is contradicted: the hits have x239 of 0 to 3.9. The GP now selects x84, x86, x150, x206 and x109, which are metal and linker RACs. Dry runs of the continued GP at kappa 0, 0.5 and 1 give nearly the same picks (1193, 3210, 5058, 5298, 613, 3430, 7318 ...), mostly from this family plus a few nearby chemistries. These results support the domain prior (small pores, strong binding sites).
**Decision:** continue label gpucb (work/gpbo.py, unchanged; saved state reused). It is the method that produced every new high, and its dry-run picks match the structure found.
**Hyperparameters:** unchanged: kappa 0.5, k_sel 5, geo5 features (Df, Di, POAVF, density, POAV), Df > 0 mask, seed 0. The dry run shows kappa barely changes the picks, so there is no reason to restart.
**Block length:** 15 measurements. That is enough to test the remaining family members and nearby chemistries, and leaves 20 for a final exploitation phase, possibly concentrating on whichever linker family proves best.
**Review:** values for the x86 = 13.36 family (4355, 4356, 3656, 2806) and the 15.65 family. Was anything above 8.66? Did the exploratory picks (1193, 5298) pay off? For the final 20: kappa 0 GP, or a fixed list of the top family members.

Proposal: `{"script": "work/gpbo.py", "label": "gpucb", "args": {"seed": 0, "k_sel": 5, "kappa": 0.5, "geo_idx": [56, 57, 61, 64, 60], "mask_col": 56}, "n_points": 15, "state": "continue", "warm_start": false}`


Runner: Block finished: 15 paid measurements; 20 remain.

## Block 6 (after 80 of 100 measurements)

**Budget:** 80 used / 100 total; 20 remaining.
**Question:** Within the low-chi-metal family (x84 = 1.72 / 1.49), which unmeasured members beat 9.32? Does high accessible surface area with low density keep raising uptake?
**Mode:** exploit. The in-loop GP refit adapts after every point.
**Gain:** a realistic chance of beating 9.32. A family-only GP predicts 8.3-9.2 ± 0.9-1.1 mol/kg for its top pick, 7475, and about 7.5-8.1 for the next several.
**Evidence:** Block 5 (continued GP-UCB) raised the best to 9.32 (cand 3656: x84 1.72, x86 13.36, ASA 2084, POAVF 0.56, density 0.83, Df 5.3). It also found 3668 = 8.02, the first member of a new linker (x86 11.69) with ASA 2174. 9 of its 15 picks scored above 6. Inside the family (20 measured), uptake rises with ASA (x69) and falls with density. Two members with matching linkers and Df 5.3-5.5 but ASA only about 1440 gave 6.9, while 3656 with ASA 2084 gave 9.3. The 15.65 linker sits about 1 mol/kg below the 13.36 linker at equal ASA. An ARD GP fitted only on the family, with features ASA, density, Df, x211 and x86, has LOO Spearman 0.81 and RMSE 1.2 on 20 points (2-feature model: 0.16). The global gpbo dry run at kappa 0 picks low-ASA 15.65 members (4161, 3563), which look like 3562/6062 (6.1-6.6), plus off-family metals (x84 1.86-1.93; the only measured off-family neighbour, 8036 at x84 1.84, gave 2.5). So it is unlikely to beat 9.32. Domain prior used: strong binding sites and small pores drive low-pressure CO2 uptake, so I keep Df at or below about 6.5 favoured and let the data decide on the larger-pore members.
**Decision:** switch to a custom family-restricted GP (work/famgp.py, check.py OK), restarted and warm-started with all 80 points. It trains only on family members and picks only family candidates with Df > 0 and ASA > 0. Label gpucb is paused, not crashed.
**Hyperparameters:** kappa 0, pure exploitation with 20 left. The dry run at kappa 0 gives 7475, 779, 6413, 5856, 1541, 919, 5058, 5920, 4034, 3210. Kappa 0.5-1 drifts to larger pores and outliers (4034 with Df 8.3; 8520). Features: ASA, density, Df, x211, x86, the set with the best family LOO. ARD Matern 2.5 + white noise, 3 restarts, seed 0.
**Block length:** 10 measurements. The ranking is fairly flat beyond the top 3-4, so I review halfway through the remaining budget and keep 10 for exploiting whatever this block finds (e.g. near-duplicates of a new best).
**Review:** did 7475, 779 or 919 beat 9.32? Does the family GP's predicted-vs-actual hold up (RMSE about 1.2)? For the last 10: continue famgp with kappa 0, or fixed points on near-duplicates of a new best.

Proposal: `{"script": "work/famgp.py", "label": "famgp", "args": {"seed": 0, "kappa": 0.0, "feat_idx": [54, 64, 56, 169, 71], "metal_col": 69, "metals": [1.72, 1.49]}, "n_points": 10, "state": "restart", "warm_start": true}`


Runner: Block finished: 10 paid measurements; 10 remain.

## Block 7 (after 90 of 100 measurements)

**Budget:** 90 used / 100 total; 10 remaining.
**Question:** Which unmeasured candidates most closely match the profile of the top cluster and could beat 9.32? The family GP's "more surface area, lower density" extrapolation failed in block 6.
**Mode:** exploit, with one diversifying analogue test.
**Gain:** a realistic chance of beating 9.32 with 7759. The global GP predicts 8.0 ± 0.6 for it, the top unmeasured prediction, and ExtraTrees also ranks it first. The other picks are predicted at 5.4-7.7, and each one also tests a sibling group I could follow up in the last 5.
**Evidence:** Block 6 (famgp, kappa 0) gave nothing above 6.70. 7475, with Di 16.8 and density 0.59, gave 1.29. 919, with Df 6.2 and ASA 2541, gave 5.0. 1541 gave 3.6. So the trend of uptake rising with surface area does not extend to larger pores. The top points (9.32, 8.66, 8.46, 8.23, 8.13) share these values: x84 1.72, x109 about 16, x211 -12, Df 4.1-5.3, Di 4.7-6.4, density 0.83-1.0 and POAVF 0.45-0.56. Members of the 13.36 linker family that remain unmeasured are either large-pore or low-ASA. Candidate 7759 (x84 1.72, x86 17.5, x109 16, x211 -12, Df 4.91, Di 6.09, POAVF 0.57, density 0.83, ASA 1979) almost copies the geometry of 3656 and has the low x109 seen at the top. 291 (GP 7.7 ± 0.9) and 5920 (GP 7.4 ± 1.2; same linker as 7318/7319, which gave 7.6-7.7) are the next GP picks. 292 is a pick of both the global gpbo and the GP. 4630 (x84 1.00) nearly copies the geometry of 6505/8469 and has x211 -12 and x239 3.91 like the top family. Measured x84 = 1.00 points reached only 5.5, but they had x211 = -20.1, which in the Mg family cost about 2 mol/kg. The global gpbo dry run (kappa 0/0.5) mostly picks x84 2.72 frameworks; that metal averages 2.2 over 15 measurements, so I rejected those picks. Domain prior used: small pores with strong binding sites favour low-pressure CO2 uptake.
**Decision:** switch to a curated fixed list (lib/fixed_points) of the 5 strongest candidates, chosen by agreement between a GP and ExtraTrees on 11 descriptors and by closeness to the top cluster's structure. No state reused.
**Hyperparameters:** points [7759, 291, 5920, 292, 4630]; no other settings.
**Block length:** 5 measurements. The last 5 can then follow whichever group succeeds: siblings 5919/6413 if 5920 is high, 7763/5860/4629 if 4630 is high, or 6822/6823/4308 for the Mg linkers.
**Review:** Did 7759 beat 9.32? Did 4630 show that low-chi x84 = 1.00 metals with x211 -12 work? Pick the final 5 by refitting work/rank7.py.

Proposal: `{"script": "lib/fixed_points.py", "label": "final_a", "args": {"points": [7759, 291, 5920, 292, 4630]}, "n_points": 5, "state": "restart", "warm_start": false}`


Runner: Block finished: 5 paid measurements; 5 remain.

## Block 8 (after 95 of 100 measurements)

**Budget:** 95 used / 100 total; 5 remaining.
**Question:** Can low-chi metals (x84 0.90-1.00) with the x211 = -12, x239 of about 3.9 linker chemistry beat 9.32? Candidate 4630 just showed they can come close.
**Mode:** exploit (final block).
**Gain:** a realistic chance of a new best. 7763 is a near-twin of 4630 (same metal, same linker RACs, Df 5.69, Di 6.83, density 0.82, ASA 1610). The GP predicts 8.97 ± 0.80 for it, with P(>9.32) about 0.33, and ExtraTrees ranks it first (7.46). The other 4 picks have P(>9.32) of 0.13-0.26 under the GP.
**Evidence:** In block 7, 4630 (x84 1.00, x211 -12, x239 3.91, Df 5.6, Di 5.8, POAVF 0.48, density 0.88) gave 9.16, the second-best value of the campaign. This confirms the hypothesis from the last call: the earlier poor results for Ca were caused by the x211 = -20.1 linker, not by the metal. The other picks were 7759 7.56, 291 5.06, 5920 3.63 and 292 4.97. The last three are Mg frameworks with x239 of 0-0.7. Across the measured low-chi, x211 of about -12 frameworks, every point at 8 or above has x239 of 1.7-3.9 (except 3429). The x239 = -2.2 framework 2251 (same metal as 4630) gave only 3.7. I refitted the GP (11 descriptors, ARD Matern) and ExtraTrees on all 95 points (work/rank8.py). The top 5 by probability of improvement are all low-chi, x211 = -12 and x239 3.6-4.3: 7763, 7326, 7760, 7869 and 2528. Domain prior used: small pores and strong open-metal binding sites favour CO2 uptake at 0.15 bar. All 5 picks keep Df at or below about 6 except 7326 (6.0), and 7760 has Mg-like geometry (Df 4.5, POAVF 0.55, density 0.98).
**Decision:** switch to a final curated fixed list (lib/fixed_points); no state reused. The picks are ranked by the GP's probability of exceeding 9.32, with ExtraTrees as a cross-check.
**Hyperparameters:** points [7763, 7326, 7760, 7869, 2528]; no other settings.
**Block length:** 5, the whole remaining budget. This is the final block, and fixed points need no adaptation.
**Review:** campaign ends; report the best of 3656 (9.32) and the new points.

Proposal: `{"script": "lib/fixed_points.py", "label": "final_b", "args": {"points": [7763, 7326, 7760, 7869, 2528]}, "n_points": 5, "state": "restart", "warm_start": false}`


Runner: Block finished: 5 paid measurements; 0 remain.
