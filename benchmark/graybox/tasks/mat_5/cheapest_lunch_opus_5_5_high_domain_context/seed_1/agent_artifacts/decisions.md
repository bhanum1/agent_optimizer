# Decisions

## Block 1 (after 0 of 100 measurements)

**Budget:** 0 used / 100 total; 100 remaining.
**Question:** No labels yet. Which regions of the MOF pool (pore size, void fraction, density, metal/linker chemistry) carry high CO2 uptake at 0.15 bar, and does the domain prior (small pores favoured, nonporous materials poor) hold?
**Mode:** both (explore with diversity, exploit with a prior-weighted half).
**Gain:** 20 labelled points spanning the pool give a basis for fitting a surrogate on a reduced feature set (geometry + RAC summaries) and to test the prior. A strong hit in the prior subset would also set a good incumbent early.
**Evidence:** Pool of 8540 MOFs, 327 descriptors (graph indices, Zeo++ geometry, RACs). No measurements. Domain knowledge (a prior, not measurements): at 298 K / 0.15 bar, gravimetric CO2 uptake is driven by adsorption strength. Ultramicro/microporous frameworks (Di ~4-7 Å), moderate void fraction, polar functional groups and strong framework electrostatics should help. Very large pores and low density should give low uptake at low pressure. Frameworks with no accessible surface (ASA=0, 33% of pool) are probably near zero. Assumptions: deterministic objective, and candidate indices are 0-based rows of candidates.csv.
**Decision:** Initial design via fixed_points (work/design.py). 10 k-means medoids over the porous pool, using 11 standardized features: Di, Df, POAVF, density, log ASA, log grav SA, log cell volume, mc-chi-0, log mc-Z-0, lc-chi-0, func-chi-0. 9 medoids within the prior subset (ASA>0, Di 4-7 Å, POAVF 0.25-0.6, density 0.7-1.7; 1815 candidates). 1 random nonporous candidate to test the "ASA=0 is poor" prior, and 1 random porous candidate with Di 3.3-4.3 Å to test the ultramicropore end.
**Hyperparameters:** k=10 and k=9 (kmeans2 ++ init, 10 restarts, seeds 0/1); random picks seeded 0 and 1.
**Block length:** 20 measurements (20% of budget). That is enough to fit a low-dimensional surrogate and to judge the prior, while leaving 80 for model-guided search.
**Review:** Check the uptake distribution, and whether the prior subset beats the diverse medoids. Check correlations of y with Di, POAVF, density, ASA and chemistry. Check whether the nonporous pick is ~0. Then choose a BO method: likely a custom GP/RF + EI on a reduced feature set over the pool, maybe restricted to ASA>0, rather than library gp_bo on all 327 dims.

Proposal: `{"script": "lib/fixed_points.py", "label": "init", "args": {"seed": 0, "points": [7690, 5323, 8005, 52, 3949, 1407, 6440, 8464, 94, 292, 3170, 5279, 5074, 2469, 4426, 1496, 2589, 1442, 7170, 4843]}, "n_points": 20, "state": "restart", "warm_start": false}`


Runner: Block finished: 20 paid measurements; 80 remain.

## Block 2 (after 20 of 100 measurements)

**Budget:** 20 used / 100 total; 80 remaining.
**Question:** Can a surrogate on a small set of physical features (metal electronegativity, pore geometry, density, cell size) steer toward high-uptake MOFs? Is the "low-electronegativity metal (lanthanide/Mg) node" signal real?
**Mode:** both (mostly exploit, guided by a model).
**Gain:** Points from model-guided EI, which concentrates on lanthanide nodes with moderate pores, the region of the current best (Nd, 6.41 mol/kg). A hit above 6.4 is plausible. The block also tests whether the lanthanide/low-chi signal generalises: if many picks land above about 3 mol/kg it is confirmed and we narrow the search; if not, we broaden the features or the method.
**Evidence:** In 20 measured points, uptake ranges from 0.09 to 6.41 mol/kg. All 3 Nd-node MOFs (metal chi 1.14) are in the top 5 (6.41, 4.54, 4.05). The Mg one is 4.97, Ga 4.55, Cu 3.55, and Zn/Co mostly 0.6-2.5. The nonporous (ASA=0) U-MOF gave 0.09, which supports prior H2. Large-pore, low-density frameworks (Di 9-22 Å) gave about 0.57, which supports prior H1. Among porous points, Spearman correlation of log uptake is +0.40 with density, -0.35 with Di and -0.34 with metal chi. The pool has about 1300 lanthanide MOFs. GP LOO is weak: Spearman 0.1-0.4 across feature sets, and rmse is about the same as the std of log uptake. So the model mainly encodes "near the best plus low chi". Domain prior used: small pores, moderate void fraction and strong framework electrostatics favour low-pressure CO2 uptake.
**Decision:** Switch from the fixed design to a custom pool GP-EI (work/poolgp.py), warm-started with all 20 points. It uses a numpy Matern-5/2 ARD GP fitted by MAP. I wrote it in numpy because botorch fitting fails in this sandbox on threadpoolctl access to /proc. Its 9 features are sqrt(mc-chi-0) (metal chi), Di, Df, POAVF, density, log grav SA, log cell volume, log num_node and D_mc-chi-1. The target is log(uptake+0.1). Offline, the first proposals are lanthanide (Yb/Tb) MOFs with moderate porosity.
**Hyperparameters:** xi=0.01 (log space, mild exploration); no greedy steps; 4 MAP restarts. Lengthscale prior lognormal around 0.3*sqrt(d) (sigma 1.2); noise prior around 1e-2 because the objective is deterministic. The features are scaled to the pool's min-max.
**Block length:** 20 measurements. That is enough to see whether EI finds better points and to recheck LOO with 40 points, while leaving 60 to adjust.
**Review:** Check whether the best improved and the hit rate above 3 mol/kg among picks, and which metals and geometries were picked. Recompute LOO for the feature sets. If lanthanides dominate, consider restricting to them or exploiting nearest neighbours of the best. If the GP stalls, try greedy steps or different features.

Proposal: `{"script": "work/poolgp.py", "label": "gpei", "args": {"seed": 0, "xi": 0.01, "greedy_every": 0, "restarts": 4}, "n_points": 20, "state": "restart", "warm_start": true}`


Runner: Block finished: 20 paid measurements; 60 remain.

## Block 3 (after 40 of 100 measurements)

**Budget:** 40 used / 100 total; 60 remaining.
**Question:** Does a surrogate that adds graph-topology descriptors (algebraic connectivity x0, Louvain community count x42) and linker chi (x164), with outlier-robust scaling, find MOFs above the 6.41 incumbent?
**Mode:** exploit (also tests the x0 hypothesis).
**Gain:** In block 2 the GP-EI ran on min-max-scaled features. Outliers compress most of the pool in that scaling, so the GP spent about 6 of 20 points on extreme structures (Di 54 Å, Di 0, density 4.5, cell with 1080 nodes), all below 0.9 mol/kg. Block 2 still found Ca 5.53 and 5.09, Sm 4.39, Er 4.30, Eu 3.66 and Sr 3.03, but nothing beat 6.41. The new model ranks much better in LOO, so it should place picks in the high-uptake region. One sequential point per family also tests whether high x0 alone, without a low-chi metal, gives high uptake.
**Evidence:** All 40 points, LOO on log(uptake+0.1):
- Old 9-feature min-max GP: Spearman 0.27, rmse 1.02.
- Same features with 1–99 percentile clipping: Spearman 0.35.
- Adding x164, log x42 and log x0 (clipped): Spearman 0.83, rmse 0.60.
- LOO top-8 picks average 4.1 mol/kg against 2.1 for all points.

Ablation shows x0 drives the gain: adding x0 alone lifts Spearman to 0.76. I chose x0 by univariate screening, so this LOO is optimistic. But its Spearman with log uptake replicates across blocks (block 1: 0.56; block 2: 0.53, on points the earlier model chose without regard to x0). Its partial correlation after removing log num_node is weak (0.20), so it mostly encodes "small, well-connected cells". All top-12 points have x0 between 110 and 205.

Low-chi metals still lead the data: Nd 6.41, Ca 5.53/5.09, Mg 4.97, other lanthanides 3.7–4.4. The fitted chi lengthscale is long (2.7), so the model now treats metal as weak. Its top proposals are a Zn family (idx 1160–1171 and 1505: Di 5.1, POAVF 0.44, density 1.2, 80 nodes, x0 about 160) with predicted mean about 10. That is an overconfident extrapolation. Measured Zn MOFs gave 0.3–2.5, though those had lower x0. One measurement from that family settles it, and the GP then updates all its near-duplicates. Domain prior used: moderate pores (Di 4.5–7 Å), moderate void fraction and dense frameworks favour uptake at 0.15 bar, and the data supports this.
**Decision:** Switch from the gpei label to work/poolgp2.py (label gpei2), a copy of poolgp.py with the new 12-feature set and percentile-clipped scaling. It restarts warm with all 40 points. The test: if the Zn high-x0 pick is high, x0/geometry dominates and metal matters less, so keep exploiting broadly. If it is low, the GP down-weights that family and should move to low-chi plus high-x0 candidates (e.g. La idx 8408, predicted 9.9). If chi still looks ignored after that, add a metal prior or restrict the pool to low-chi metals.
**Hyperparameters:** xi=0.01 (EI in log space; mu already exceeds the incumbent, so this is effectively exploitative). 4 MAP restarts. Same lengthscale and noise priors as poolgp (deterministic objective). The check.py run passed.
**Block length:** 15 measurements. That is enough for the GP to test the Zn family, move on, and exploit the low-chi/high-x0 region, while leaving 45 to re-steer (metal restriction, local neighbourhood exploitation).
**Review:** Did the best improve? Count picks above 4 mol/kg. Check the measured value of the Zn high-x0 pick and the metals picked. Refit LOO for "plus clip" against plus a metal-restricted variant. Check whether predictions of 10 shrank to realistic values.

Proposal: `{"script": "work/poolgp2.py", "label": "gpei2", "args": {"seed": 0, "xi": 0.01, "greedy_every": 0, "restarts": 4}, "n_points": 15, "state": "restart", "warm_start": true}`


Runner: Block finished: 15 paid measurements; 45 remain.

## Block 4 (after 55 of 100 measurements)

**Budget:** 55 used / 100 total; 45 remaining.
**Question:** Does a surrogate that knows the metal family (alkaline earth vs lanthanide vs other), searching only plausible porous geometries, find MOFs above the new 8.04 incumbent?
**Mode:** exploit, which also tests whether the alkaline-earth signal holds.
**Gain:** The current gpei2 model has failed. With the 8.04 incumbent, its EI now goes to Di=0 / huge-cell outlier structures with predicted mean about 1.2 and large sd. Continuing it would waste the block. The new model ranks much better in LOO and points EI at Ca/Sr/Mg frameworks like the incumbent, so we can plausibly hope for 1–3 more points above 6 and a chance to beat 8.04.
**Evidence:** Block 3 found a new best on its first pick: Sr idx 7760, 8.04 mol/kg (Di 6.1, POAVF 0.55, density 0.98, 57 nodes, x0 148). The rest of the block was mediocre: Zn high-x0 1170 gave 1.15, which refutes "x0 alone". Lanthanides gave 1.6–3.5, Na 0.02/1.75, and nonporous Ca 2.95.

Porous alkaline-earth MOFs so far: 8.04, 5.53, 5.09, 4.97, 3.03 and 1.46 (median about 5). Porous lanthanides: median about 2.4. Transition metals are mostly below 2.5.

LOO over all 55 points (work/loo6.py):
| Feature set | Spearman | rmse | Mean uptake of LOO top-10 |
|---|---|---|---|
| v2 "plus" 12 features | 0.21 (was 0.83 at 40 pts, x0 effect was overfit) | 1.10 | 2.5 |
| plus + alkaline-earth/lanthanide indicators | 0.66 | 0.77 | 3.6 |
| compact 8 features | 0.54 | — | — |

The indicators were chosen after looking at the data, so these numbers are optimistic. They agree with domain chemistry: hard, low-electronegativity Ca/Sr/Mg carboxylate nodes give strong electrostatics with CO2. Restricting the search to ASA>0, Di 3.8–8.5, POAVF 0.2–0.65, density 0.7–2.2 and ≤400 nodes leaves 2782 candidates. Every point above 3 mol/kg except the nonporous Ca pair satisfies it. Offline, the top EI picks are Mg 1193, Ca 2778/2779/7763/4631, Sr 4723/4314 and K 3299, with predicted means of 6–7.7. Those means are optimistic.
**Decision:** Switch to work/poolgp3.py, label gpei3: the poolgp2 GP plus two metal-family indicator features, with the restricted pool. It is a restart with warm start on all 55 points. If it hits Ca/Sr picks above 5, the alkaline-earth hypothesis is confirmed and the last ~30 measurements go to exploitation (mean-greedy) around the best family. If those picks come in below 3, the metal signal is overfit, and I fall back to local neighbours of 7760/4426.
**Hyperparameters:** xi=0.01 (log space, nearly pure EI; uncertainty in the restricted pool is moderate). 4 MAP restarts. Same priors as poolgp2. restrict=True removes the outlier regions that drew EI in blocks 2 and 3. The check.py run passed.
**Block length:** 15 measurements. That is enough to test about 6–10 alkaline-earth candidates and let the GP adapt, while leaving 30 for final exploitation.
**Review:** Check whether the best improved and how many picks exceeded 5. Measured values by metal family. Whether EI drifts to Zn/Cu (chi treated as weak). Recheck LOO. Then plan the final greedy/local exploitation.

Proposal: `{"script": "work/poolgp3.py", "label": "gpei3", "args": {"seed": 0, "xi": 0.01, "greedy_every": 0, "restarts": 4, "restrict": true}, "n_points": 15, "state": "restart", "warm_start": true}`


Runner: Block finished: 15 paid measurements; 30 remain.

## Block 5 (after 70 of 100 measurements)

**Budget:** 70 used / 100 total; 30 remaining.
**Question:** Can a GP that searches only alkaline-earth (Be/Mg/Ca/Sr/Ba) frameworks of plausible porous geometry beat the 8.04 incumbent (Sr idx 7760)? Block 4 confirmed that metal family is the main signal.
**Mode:** exploit (mostly), with EI keeping some upside-seeking exploration.
**Gain:** In block 4 the alkaline-earth picks averaged 4.7 mol/kg; 5 of 9 were above 4.9, with the best at 6.78. The 6 non-alkaline-earth picks averaged 1.5. Restricting to the 103 unmeasured alkaline-earth candidates removes the lanthanide and transition-metal picks that the global EI now favours. We can hope for several picks of 5–7 and a fair chance of one above 8.04.
**Evidence:** Block 4 (gpei3) made no new best, but gave Mg 6629 6.78, Mg 4161 6.30, Ca 3130 5.92, Mg 5353 5.19, Mg 5027 4.90, Ca 4631 4.41 and Mg 1193 3.93. The misses: Ca 2777 1.84 (Di 7.95), Sr 3536 3.31, K 0.92, Zn 1.89, Cd 1.51, Ce 0.91, Co 0.47.

Calibration: I fitted on the first 55 points and predicted the 15 block-4 points. Spearman was only 0.10, so within the alkaline-earth family the GP ranks poorly and its predicted means are poorly calibrated (pred 7.3 vs actual 3.9; pred 3.4 vs actual 5.2). Its family-level signal (alkaline earth vs others) is right, and the domain chemistry agrees: hard, low-electronegativity alkaline-earth carboxylate nodes give strong electrostatics with CO2.

On all data, the unrestricted EI now prefers lanthanide families (Ln 1962–1966, 7891–7894, 553) with predicted means about 6 and large sd. Measured porous lanthanides have a median of about 2.4 (max 6.41, from one of about 20), so I down-weight them. The alkaline-earth pool in the restricted geometry has 103 unmeasured candidates: 66 Mg, 23 Ca, 11 Sr, 2 Ba, 1 Be. Alkaline-earth candidates outside the geometry restriction are mostly nonporous, and nonporous Ca measured only 2.8–2.95, so they stay excluded. The first offline picks are Ca 1251, then 4424 and 6504.
**Decision:** Switch to work/poolgp4.py, label gpei4. It is a copy of poolgp3 (same 14 features, GP and priors) plus a `metals` filter restricting the pool to alkaline earths, with the same geometry restriction. It is a restart with warm start on all 70 points (the model is refitted every ask anyway, so nothing is lost versus continuing).
**Hyperparameters:**
- xi=0.01: EI in log space.
- greedy_every=3: every third pick is the posterior-mean argmax. The within-family ranking is weak, so mixing EI (sd-driven, explores the 103 candidates) with mean picks (exploits near the Mg/Ca/Sr hits) hedges between them.
- restarts=4.
- metals=[4,12,20,38,56] and restrict=True.

check.py passed.
**Block length:** 15 measurements, which covers about 15% of the alkaline-earth pool. That leaves 15 for a final block, which can be pure greedy within alkaline earths, or local neighbours of the best, depending on what this block shows.
**Review:** Check whether the best improved and how many picks exceeded 5, by Mg, Ca and Sr. Check whether the GP's within-alkaline-earth ranking improved (compare predicted and actual values for this block). Then decide the final 15: greedy mean in alkaline earths (greedy_every=1), or nearest neighbours of the top points, possibly with a few top lanthanide-EI picks if the alkaline-earth pool looks exhausted.

Proposal: `{"script": "work/poolgp4.py", "label": "gpei4", "args": {"seed": 0, "xi": 0.01, "greedy_every": 3, "restarts": 4, "restrict": true, "metals": [4, 12, 20, 38, 56]}, "n_points": 15, "state": "restart", "warm_start": true}`


Runner: Block finished: 15 paid measurements; 15 remain.

## Block 6 (after 85 of 100 measurements)

**Budget:** 85 used / 100 total; 15 remaining.
**Question:** Which of the remaining porous alkaline-earth (AE) candidates are most likely to beat the new best of 9.16 (Ca idx 4630)?
**Mode:** exploit.
**Gain:** In block 5 the AE-only GP found 10 of 15 picks above 6 mol/kg. The new best is Ca 4630 at 9.16, with 8.66, 8.46, 8.23, 8.02 and 7.80 behind it. The 13 measured points inside the "sweet spot" rule (defined under Decision) average about 7. Picking the best-ranked candidates from that region should give several more points of 6–9, with a real chance of beating 9.16.
**Evidence:**
- Block 5 GP calibration: I fitted on blocks ≤4 and predicted the block-5 points (work/cal6.py). Spearman was only 0.20. The model badly under-predicted the big hits (pred 1.9 for actual 8.66, pred 2.9 for 8.23). So the GP mean is a weak ranker inside the AE family, and the family/geometry filter did most of the work.
- A simple 4-nearest-neighbour model (distance-weighted, on Di, POAVF, density and fchi, over the 30 measured porous AE points) has leave-one-out Spearman 0.62 (p=3e-4; work/r6b.py).
- Strong pattern: Mg/Ca/Sr with POAVF 0.45–0.6, density 0.76–1.0 and Di 4.5–6.5 gave 7–9.2 (one exception, 3.93). Mg with POAVF 0.35–0.37 gave 6.1–6.6. Di above 7 failed (2.79, 1.84, 4.28). Be gave 0.83.
- Domain prior (hard, low-electronegativity AE carboxylate nodes with moderate pores give strong CO2 electrostatics) agrees with this.
- 30 unmeasured AE candidates fall in the relaxed region (POAVF 0.36–0.66, density 0.7–1.1, Di 4.2–7.0, ASA>0).
**Decision:** Switch from the GP (gpei4) to a fixed list of the 10 candidates ranked highest by the average of the kNN and GP predictions, favouring higher POAVF:
- Mg 3656, 6504, 3210, 7319, 4355, 7634, 3212, 919;
- Ca 4629 (index next to the best, 4630);
- Sr 4314 (geometry like the earlier Sr 8.04).

I dropped near-duplicate 7318 (twin of 7319) so the picks stay diverse.
**Hyperparameters:** Not applicable (fixed_points). The rankings come from work/r6b.py: kNN k=4, weights 1/(d+0.3); the GP is poolgp3 on all 85 points.
**Block length:** 10 measurements. That keeps 5 for a final block that can react to these results: neighbours of any new high point, or the next-ranked candidates (7318, 2806, 7531, 7763, 2778, 4721, 5058).
**Review:** Did any pick beat 9.16? Was the hit rate above 6 consistent with the rule? Refit kNN/GP with the new data and pick the final 5 from the remaining region candidates or near-twins of the best new points.

Proposal: `{"script": "lib/fixed_points.py", "label": "knn6", "args": {"points": [3656, 6504, 3210, 7319, 4355, 7634, 3212, 919, 4629, 4314]}, "n_points": 10, "state": "restart", "warm_start": false}`


Runner: Block finished: 10 paid measurements; 5 remain.

## Block 7 (after 95 of 100 measurements)

**Budget:** 95 used / 100 total; 5 remaining.
**Question:** Which 5 unmeasured porous alkaline-earth (AE) candidates are most likely to beat the new best of 9.32 (Mg idx 3656)?
**Mode:** exploit (final block).
**Gain:** In block 6, 6 of 10 picks scored above 6, with a new best of 9.32 (Mg 3656, up from 9.16) and 8.13 (Mg 6504). These five picks are the highest-ranked candidates left in the sweet spot. Each has a realistic chance of reaching 7–9 and a small chance of beating 9.32.
**Evidence:**
- After 95 points, the 4-nearest-neighbour model (on Di, POAVF, density, fchi) still has leave-one-out Spearman 0.54 (p=4e-4) on the porous AE points.
- Within AE, POAVF (+0.26), ASA (+0.27) and density (−0.24) correlate most with uptake.
- Di above 6.5 is risky: 919 gave 4.99, 292 gave 4.97, 6823 gave 2.79.
- The nearest descriptor neighbours of 3656 are Co, Zn or Cu frameworks, which have measured poorly, so I do not chase them.

The picks:
- 7318: near-twin of 7319 (7.68), with POAVF 0.53 and density 0.93.
- 7763: Ca, kNN prediction 7.56, POAVF 0.52, density 0.82.
- 2970: Mg, kNN 7.54, geometry between 3656 (9.32) and 3668 (8.02).
- 5058: Mg, POAVF 0.48, density 0.85.
- 5767: Sr, Di 5.38, density 1.02; an earlier Sr scored 8.04.
**Decision:** Fixed points from a ranking that combines kNN and GP predictions with the domain rule (AE nodes; moderate pores, POAVF about 0.45–0.6, density 0.75–1.0). Diversified across Mg, Ca and Sr.
**Hyperparameters:** Not applicable (fixed_points).
**Block length:** 5 measurements, the whole remaining budget.
**Review:** Campaign ends; the final result is the best measured value.

Proposal: `{"script": "lib/fixed_points.py", "label": "final7", "args": {"points": [7318, 7763, 2970, 5058, 5767]}, "n_points": 5, "state": "restart", "warm_start": false}`


Runner: Block finished: 5 paid measurements; 0 remain.
