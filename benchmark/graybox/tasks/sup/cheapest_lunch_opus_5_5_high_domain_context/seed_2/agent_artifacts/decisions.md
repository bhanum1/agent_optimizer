# Decisions

## Block 1 (after 0 of 250 measurements)

**Budget:** 0 used / 250 total; 250 remaining.
**Question:** Does the domain prior (cuprates, especially Hg/Tl/Bi families, have the highest Tc) hold in this pool, and which cuprate family/region is best to concentrate model-guided search on?
**Mode:** both (domain-informed initial design; most points are in families expected to contain the top Tc).
**Gain:** 24 labelled points spread over the families, giving a first incumbent (likely ~100 K+ if the prior holds) and training data for a pool surrogate. If Hg/Tl beat others, the next block restricts BO to those; if not, widen to all cuprates.
**Evidence:** No measurements yet. Pool: 21263 candidates, 81 composition descriptors (UCI-style elemental property statistics), 6093 duplicate feature rows. Domain prior (chemistry knowledge, not measurement): cuprates give the highest Tc; O has thermal conductivity 0.02658 and Cu ~400, so range_ThermalConductivity (x67) = 399.97 flags Cu–O compounds (10419 candidates). Within those, range_atomic_mass (x7) identifies the heaviest element vs O: 184.59 = Hg (571), 188.38 = Tl (613), 192.98 = Bi (1687), 121.33 = Ba-max e.g. YBCO (2059). Known Tc ordering Hg > Tl > Bi > YBCO > La-type.
**Decision:** experiment/initial design with fixed_points: k-means (scipy kmeans2, standardized features, deduplicated) representatives: Hg 7, Tl 6, Bi 4, Y/Ba 3, other cuprates 2, non-cuprates 2 (cheap check of the prior and contrast for a surrogate).
**Hyperparameters:** not applicable (fixed list, work/design1.py).
**Block length:** 24 measurements (~10% of budget), enough for per-family means and a first surrogate fit.
**Review:** per-family Tc; whether non-cuprates are low; fit GP on cuprate subset (LOO quality); then build a custom pool GP-EI optimizer restricted to the promising family subset.

Proposal: `{"script": "lib/fixed_points.py", "label": "design1", "args": {"seed": 0, "points": [3666, 7793, 5311, 3639, 17964, 7786, 3896, 19015, 6202, 7040, 2696, 9026, 3265, 3270, 4765, 815, 2086, 1788, 1164, 8978, 7673, 5169, 20715, 15160]}, "n_points": 24, "state": "restart", "warm_start": false}`


Runner: Block finished: 24 paid measurements; 226 remain.

## Block 2 (after 24 of 250 measurements)

**Budget:** 24 used / 250 total; 226 remaining.
**Question:** Where inside the Hg-cuprate family (best so far) are the highest-Tc compositions, and does a subset GP learn within-family structure?
**Mode:** exploit (with a learning check on the within-family model).
**Gain:** Points near the 132 K Hg-1223-like incumbent (7786) and the 128.5 K 6-element Hg point (3896); realistic hope 135-140 K. Also ~20 more Hg labels to test whether the GP gains predictive skill (LOO) — if not, switch to pure neighbourhood/prior-based picking.
**Evidence:** Block 1 per family (Tc, K): Hg 45, 89.5, 89, 115, 104.8, 132, 128.5 (mean 100, max 132); Tl 100.4, 108, 65, 50, 42.1, 104 (max 108); Bi 89, 109, 85, 7; Y/Ba 74.9, 76.1, 113; other cuprates 46, 9.6; non-cuprates 0.76, 0.47. Domain prior (Hg > Tl > Bi > YBCO > non-cuprates) confirmed in ordering of maxima. 7786 has wtd-mean mass 54.0 consistent with HgBa2Ca2Cu3O8 (Hg-1223, the known record family); 46 candidates lie within 0.5 z-distance of it. Offline: GP on 22 cuprate points has no LOO skill (corr ~0, rmse ~40 K), so the GP's role now is mostly EI-driven local exploitation around the top points (top picks are neighbours of 3896/7786 with predicted ~135 K). Botorch fit fails in this sandbox (threadpoolctl /proc), so I wrote a pure numpy/scipy GP.
**Decision:** switch to a custom pool optimizer work/subgp.py: candidates = Hg cuprates (x67≈399.97 & x7≈184.59, 571, 7 measured); trained on all measured cuprates; features z-scored on cuprates then PCA(15) fit on the Hg subset; Matern-5/2 ARD GP (MLE w/ weak priors, 4 restarts), EI. Warm start with all 24 points. check.py passed (warm 24).
**Hyperparameters:** n_pc=15 (8 or 4 PCs made picks drift to mediocre ~82 K-predicted regions; 15 picks neighbours of the top points); EI (exploitative, appropriate as prior already identifies the family); seed 0.
**Block length:** 20 measurements: enough Hg labels to re-check GP skill and the local Tc distribution near Hg-1223 before deciding whether to widen (Tl) or tighten (kNN around best).
**Review:** best value change; LOO skill of GP within Hg with ~27 Hg points; whether picks cluster near 7786/3896 and how Tc varies there; consider adding Tl family (188.38) or UCB if EI stalls.

Proposal: `{"script": "work/subgp.py", "label": "hggp", "args": {"seed": 0, "families": [184.59], "n_pc": 15, "acquisition": "ei"}, "n_points": 20, "state": "restart", "warm_start": true}`


Runner: Block finished: 20 paid measurements; 206 remain.

## Block 3 (after 44 of 250 measurements)

**Budget:** 44 used / 250 total; 206 remaining.
**Question:** Do exact-duplicate feature rows of the top Hg compositions (repeated literature measurements of the same composition, e.g. under pressure/different samples) carry Tc above the 135 K incumbent, and how large is the within-composition spread?
**Mode:** both (exploit duplicates of the best compositions; measure within-group spread to decide whether to exhaust them).
**Gain:** Domain prior: HgBa2Ca2Cu3O8 (Hg-1223) has Tc 133-135 K at ambient and up to ~150-164 K under pressure; Hg-1212 ~127 K ambient, ~150 K under pressure. If any duplicate reports such values, best jumps 10-30 K. If spread is small, we learn duplicates are near-identical and return to model-guided search over distinct compositions.
**Evidence:** Block 2 (subgp EI, Hg family) raised best 132 -> 135 K (5318, 3373) and 133 K (4552); 10 of 20 picks were >=125 K. GP LOO within cuprates now corr 0.51 (8 PCs), rmse 30 K; its current top picks predict mean <=131 K, i.e. little expected gain. Duplicate analysis (work/an4.py): Hg family has 73 exact-duplicate groups. A 21-member group (5 elements, wtd-mean mass 54.6 = computed HgBa2Ca2Cu3O8 value) is unmeasured; a 16-member group (wm 61.5 = HgBa2CaCu2O6, Hg-1212) unmeasured; 3373 (135 K) has 8 unmeasured duplicates. Measured near-duplicate pairs show spread 0.4-33 K (3892/5341 127.8/127.4; 5317/5318 128/135; 3684/3686 128/95). The composition identification uses the domain context (descriptor names) plus chemistry, a prior not a measurement.
**Decision:** experiment/exploit with fixed_points: 10 random members of the Hg-1223 duplicate group, 4 of the Hg-1212 group, 4 duplicates of 3373 (random choice, seed 3, work/pick3.py). hggp state (subgp) is kept for later continuation/restart.
**Hyperparameters:** not applicable (fixed list).
**Block length:** 18: enough draws per group to see the spread (10 for the most promising group) before deciding whether to exhaust the remaining 11+12+4 duplicates or return to GP.
**Review:** Tc distribution per group; if any >135 or spread wide, measure remaining members of that group (and check other big dup groups); if all ~equal, go back to subgp (8 PCs, noise-aware EI) on distinct compositions, maybe adding Tl family.

Proposal: `{"script": "lib/fixed_points.py", "label": "dups3", "args": {"seed": 3, "points": [5655, 5315, 7105, 2987, 4709, 2986, 2804, 5789, 19622, 3002, 8037, 2991, 3678, 3669, 4579, 5339, 3891, 7642]}, "n_points": 18, "state": "restart", "warm_start": false}`


Runner: Block finished: 18 paid measurements; 188 remain.

## Block 4 (after 62 of 250 measurements)

**Budget:** 62 used / 250 total; 188 remaining.
**Question:** Do the remaining exact duplicates of the two best Hg compositions (Hg-1223 group, 3373 group) hold a Tc above the 135.7 K incumbent?
**Mode:** exploit (finishing the duplicate groups also tells us each group's full Tc range).
**Gain:** Hg-1223 group: 10 of 21 measured, Tc 115, 118, 120, 130.9, 132, 132.5, 133, 134.5, 135.5, 135.7. That gives a 11/21 ≈ 52% chance that the group maximum is among the 11 unmeasured. Domain prior: Hg-1223 reaches ~138 K at ambient pressure and up to ~150-164 K under pressure, so the upside could be large. 3373 group: 5 of 9 measured (135, 134.5, 130, 110.7, 110.3), 4 left.
**Evidence:** Block 3 raised the best from 135.0 to 135.7 K (7105). Within-group spread is wide (115-135.7 for Hg-1223), so duplicates act as fresh draws. The Hg-1212 group was lower (92-122), so I drop it. The GP's top distinct picks predict <=131 K, below what duplicates offer. The Tl family's nearest analog is Tl-1223, 8 dups (wm 52.6) such as 6092 and 4097. The Tl family max so far is 108, and the prior Tc for Tl-1223 is <=133, so that group has lower priority. Hg-1223 near-duplicates are 7781-7789, 7124-7126, 3979, 8358-8359 and 3426, at z-distance <0.21, probably O-content variants. 7786 among them gave 132.
**Decision:** fixed_points on all 11 remaining Hg-1223 exact duplicates and the 4 remaining 3373 duplicates.
**Hyperparameters:** not applicable.
**Block length:** 15: finishes both groups, so the next decision has their complete distributions.
**Review:** Whether the best improved. If Hg-1223 values >136 appear, do the near-duplicate O-variants (~15 pts) next. Otherwise do near-duplicates of the top points, then return to subgp (n_pc=8, noise-aware) on Hg ± Tl distinct compositions.

Proposal: `{"script": "lib/fixed_points.py", "label": "dups4", "args": {"seed": 4, "points": [3028, 3060, 3603, 4035, 4412, 4904, 5654, 5788, 6138, 7791, 8535, 3381, 4551, 4580, 5340]}, "n_points": 15, "state": "restart", "warm_start": false}`


Runner: Block finished: 15 paid measurements; 173 remain.

## Block 5 (after 77 of 250 measurements)

**Budget:** 77 used / 250 total; 173 remaining.
**Question:** Do Hg-1223 analogues with a partial cation substitution (Hg-Tl, Hg-Pb, Hg-Re) beat the 135.7 K incumbent? The Hg-Ba-Ca-Cu-O exact duplicates are now used up.
**Mode:** exploit, which also tells us which substitution family is worth exhausting.
**Gain:** Domain prior: (Hg0.8Tl0.2)Ba2Ca2Cu3O8+d holds the ambient-pressure record of about 138 K. Hg-Pb-1223 and Hg-Re-1223 are reported at about 130-135 K. A realistic upside is +1 to 3 K. Whichever subfamily scores high, its remaining members will be the next block.
**Evidence:** Block 4 finished the duplicate groups. The 21 Hg-1223 duplicates gave 107-135.7 K, and the 9 members of the 3373 group gave 110-135 K, so the best did not improve. New analysis (work/an7.py, an8-10.py) recovers each candidate's element set by matching the unweighted mean and geometric mean of atomic mass to element combinations. That gives 966 element-set groups among cuprates. Ca-Ba-Hg (119 candidates, 39 measured, mean 123, max 135.7) is the best. The F-Ca-Ba-Hg group (14 candidates) has 5 measured, mean 129, max 135. Ca-Ba-Hg-Tl has 41 candidates and only 1 measured (108 K, at wm 108, a Tl-rich phase). Its members 4029-4034 (wm 54.64-54.84) sit right next to Hg-1223 (wm 54.63), which fits Hg1-xTlx-1223 with small x. 7468, 19039 and 7469 (wm 51-54) are other Hg/Tl-1223 variants. Ca-Ba-Hg-Pb has 13 distinct compositions near wm 54 and Ca-Ba-Hg-Re has 8, all unmeasured. Element identification is an inference from descriptors and is not a measurement. Some matches are ambiguous with exotic alternatives, but the cuprate context makes the Hg reading by far the most plausible. The GP's best distinct picks predicted <=131 K, so these chemistry-guided picks are the better bet.
**Decision:** fixed_points over all 9 Hg-Tl-1223-like compositions, 6 of the 11 Hg-Pb-1223 compositions near wm 54 (alternate members of each substitution series), and 5 of the 8 Hg-Re compositions.
**Hyperparameters:** not applicable (fixed list).
**Block length:** 20. That is enough for a per-subfamily mean and max, so the next block can exhaust the winner (remaining Pb 5, Re 3 plus duplicates, Hg-1223 O-variants ~20, fluorinated 9) or fall back to a GP.
**Review:** Per-subfamily Tc. If any value is above 135.7, measure that subfamily's remaining members and duplicates. Otherwise go to the Hg-1223 O-variants (7781-7789, 7124-7126, 3979, 8358/9, 3426, 5473-5477, 5424) and the remaining fluorinated members.

Proposal: `{"script": "lib/fixed_points.py", "label": "subst5", "args": {"seed": 5, "points": [4029, 4030, 4031, 4032, 4033, 4034, 7468, 19039, 7469, 3980, 3982, 3984, 4413, 4415, 4417, 5487, 8591, 8594, 8596, 5790]}, "n_points": 20, "state": "restart", "warm_start": false}`


Runner: Block finished: 20 paid measurements; 153 remain.

## Block 6 (after 97 of 250 measurements)

**Budget:** 97 used / 250 total; 153 remaining.
**Question:** Do the closest chemical relatives of the new best point, 19039 (Hg-Tl-Ba-Ca-Cu-O, 137.4 K), or the remaining untested Hg-1223-like compositions, go above 137.4 K?
**Mode:** exploit. It also maps which Hg-1223 variants are still worth exhausting.
**Gain:** Block 5 raised the best from 135.7 to 137.4 K at 19039. That point is close to the domain-prior record of (Hg,Tl)-1223 at about 138 K. Its direct relatives are a likely place for +0.5 to 3 K. Under the prior, pressure or fluorinated entries could reach 140 K or more, and those would sit among Hg-1223 variants.
**Evidence:** Block 5 results:
- Hg-Tl 4029-4034 (wm 54.6-54.8): 132-134 K.
- 19039: 137.4 K. 7468: 110 K. 7469: 90 K.
- Hg-Pb: 103-130 K.
- Hg-Re: 130-133.1 K. That family has no upside, so its 9 remaining members are deprioritized.

New analysis:
- 19040-19043 and 19044-19047 come right after 19039 in the index, likely the same source. Their atomic masses match Hg-Tl-Ba-Ca-Cu-O plus Sm or plus Yb (n=7, wm 52.7-53.7), which fits a rare-earth-doped (Hg,Tl)-1223 series. They are the nearest relatives of the best point.
- The top two non-pure points, 19039 and F-Hg-1223 5318, both have wm 52.53. Unmeasured Ca-Ba-Hg O-variants sit at wm 52.2-52.4 (5424, 5473-5477).
- Ca-Ba-Hg dup groups: wm 84.67 gave a max of 135. The wm 74.56 (9 members) and wm 78.7 (6 members) groups are untested. 5319 and 5321 come from the same index range as the F-Hg-1223 points 5317-5322 (128-135 K).
- Untested single-dopant Hg-1223 compositions at wm 52-54.5: Nd 8249, Sb 6130/10720, Bi 3693, Mo 3682, V 3680.

Element sets are inferred from the atomic-mass descriptors (work/an14.py, an16.py). That is an inference, not a measurement.
**Decision:** use fixed_points on 23 picks:
- all 8 Sm- and Yb-doped (Hg,Tl) relatives;
- 3 members each of the wm 74.56 and wm 78.7 Ca-Ba-Hg dup groups (including 5319 and 5321);
- 3 Hg-1223 O-variants at wm 52.2-52.4;
- 6 single-dopant Hg-1223 compositions.

The subgp state is kept as a fallback. Its picks predicted at most 131 K, so I am not using it now.
**Hyperparameters:** not applicable (fixed list).
**Block length:** 23. That covers each hypothesis family with enough points to judge it. About 130 measurements then remain to exhaust whichever family wins: the rest of the 74.56/78.7 dups, the O-variants (about 17), Sr-Re/Sr, the F-group remainder, or the Hg-1212 dups.
**Review:** Which family gave values at or above 135 K. Exhaust that family and its exact duplicates. If nothing beats 137.4 K, go to the remaining Ca-Ba-Hg O-variants (wm 53.7-54.24), Hg-Sr, Sr-Re, F remainder, and Hg-Tl 1212-like 3608-3610/7467.

Proposal: `{"script": "lib/fixed_points.py", "label": "rel6", "args": {"seed": 6, "points": [19040, 19041, 19042, 19043, 19044, 19045, 19046, 19047, 5321, 5429, 17216, 5319, 4419, 4710, 5424, 5473, 5477, 8249, 6130, 10720, 3693, 3682, 3680]}, "n_points": 23, "state": "restart", "warm_start": false}`


Runner: Block finished: 23 paid measurements; 130 remain.

## Block 7 (after 120 of 250 measurements)

**Budget:** 120 used / 250 total; 130 remaining.
**Question:** Can any of the untested HgBa2Ca2Cu3O(8+d) oxygen/stoichiometry variants, or the close (Hg,Re)/(Ba,Sr), (Hg,Sb,Pb) and Hg-Sr 1223 analogues, beat 137.4 K?
**Mode:** exploit. It also finishes mapping the Hg-1223 composition family.
**Gain:** Realistic upside is +0.5 to 3 K. Under the domain prior, the Hg-1223 family holds the ambient record (about 138 K) and the high-pressure entries (up to about 164 K). Any such entry would be one of the remaining Hg-1223 compositions.
**Evidence:** Block 6 gave no improvement.
- The Sm- and Yb-doped (Hg,Tl) relatives of 19039 scored 90-124.5 K.
- Ca-Ba-Hg duplicate groups: wm 74.56 gave 98-107 K and wm 78.7 gave 111-123 K. Both are deprioritized.
- Single-dopant Hg-1223 compositions scored 120-132 K.
- The Hg-1223 O-variants were the best group of the block: wm 52.2-52.4 gave 132.9, 134.7 and 133.4 K. The earlier 7786 (wm 54.03) gave 132 K.

Untested Ca-Ba-Hg 5-element compositions at wm 52.2-54.24 (n=5, the same element set as Hg-1223): 5474-5476 lie between 5473 and 5477, which are the strongest of the new results. The others are 3426, 8358, 8359, 7124-7126, 3979 and 7781-7789.

Element sets are inferred from atomic-mass descriptors (work/an14.py, an17.py). That is an inference, not a measurement.
**Decision:** use fixed_points on 26 picks:
- all 18 remaining Hg-1223 O-variants;
- (Hg,Re)-(Ba,Sr)-1223-like 9335-9337 (wm 52.8-53.4);
- Hg-Sb-Pb 10721-10723 (wm 53.9-54.4);
- Hg-Sr 3698 (wm 54.42) and 8422.
**Hyperparameters:** not applicable (fixed list).
**Block length:** 26. That finishes the highest-mean remaining family in one pass. About 104 measurements stay available to exhaust any family that scores 135 K or more.
**Review:** If any family scores 135 K or more, exhaust its neighbours (index-adjacent, same wm). Otherwise the next candidates are:
- the F-Hg remainder (3893-3898, 3927, 5320, 5322, 5342);
- Hg-Tl 3608-3610 and 7467;
- the rest of Hg-Sr and Sr-Tl;
- the Hg-Au and Hg-Fe remainders;
- the subgp fallback.

Proposal: `{"script": "lib/fixed_points.py", "label": "ovar7", "args": {"seed": 7, "points": [5474, 5475, 5476, 3426, 8359, 7787, 7788, 7789, 8358, 7124, 7125, 3979, 7784, 7785, 7126, 7781, 7782, 7783, 9335, 9336, 9337, 10721, 10722, 10723, 3698, 8422]}, "n_points": 26, "state": "restart", "warm_start": false}`


Runner: Block finished: 26 paid measurements; 104 remain.

## Block 8 (after 146 of 250 measurements)

**Budget:** 146 used / 250 total; 104 remaining.
**Question:** Now that the Hg-1223 O-variant family is used up, which of the remaining untested families with upside could still beat 137.4 K? The candidates are Hg-1234, the Hg-1212 duplicates (where a high-pressure entry could sit), the 1223/1212-like Hg-Tl, Sr-Tl and Hg-Sr compositions, and the unidentified 1223-like multi-element entries.
**Mode:** exploit, with some tail search.
**Gain:** The chance of beating 137.4 K is small (+0 to 3 K). It also settles which remaining families are worth finishing with the last 72 points.
**Evidence:** Block 7 (26 points) gave no improvement.
- Hg-1223 O-variants scored 133-135.8 at wm 52.2-54.0 and 94-110 at wm 54.1-54.24.
- Sr-Re scored 128.7-131.3; Sb-Pb 127-133; Hg-Sr 3698 133.5; 8422 126.
- All Ca-Ba-Hg 5-element compositions at wm 52.2-54.63 have now been measured. Best overall is still 19039 (137.4 K).

What is left untested:
- **Hg-1234 (wm 50.48):** 4 entries. My mass check confirms HgBa2Ca3Cu4O10 gives wm 50.48. The kNN surrogate predicts about 134.6 K; the domain prior puts Hg-1234 at about 125-127 K ambient.
- **Hg-1212 at wm 61.53:** 12 exact duplicates. The 4 measured scored 92-122. Duplicates spread by up to about 25 K, and the domain prior says Hg-1212 under pressure reaches about 150 K.
- **(Hg,Tl) compositions:** 7467 and 3608-3610 at wm 58.5, and 9553 at wm 61.6.
- **Sr-Tl:** 3604-3606 and 8423 at wm 55.5.
- **Hg-Sr:** 3699, 3700 and 3571.
- **Unidentified 1223-like cuprates:** 19616-19617 at wm 52.0-52.6, index-adjacent to Hg-1223 entry 19622; and 9550-9551 at wm 52.0-52.4.

Element-set assignments are inferred from atomic-mass descriptors, not measured.
**Decision:** use fixed_points on 32 picks:
- Hg-1234: 3834, 4106, 5050, 7858;
- all 12 remaining Hg-1212 wm 61.53 duplicates;
- Hg-Tl: 7467, 3608-3610, 9553;
- Sr-Tl: 3604-3606, 8423;
- Hg-Sr: 3699, 3700, 3571;
- 19616, 19617, 9550, 9551.
**Hyperparameters:** not applicable (fixed list).
**Block length:** 32. That covers every remaining plausible family in one pass and leaves 72 to exhaust whichever family shows a tail at or above 135 K.
**Review:** Did any pick reach 135 K or more?
- If one did, exhaust its relatives: the Hg-1212 O-variants 7770-7780 and 17971-17973, the remaining Tl family members, or neighbours 19618-19620.
- Otherwise spend the rest on the Hg-Re wm 54.47 duplicates (10207-10209, 9273/4), the F remainder, the Hg-Au and Hg-Fe remainders, and kNN or subgp picks.

Proposal: `{"script": "lib/fixed_points.py", "label": "tail8", "args": {"seed": 8, "points": [3834, 4106, 5050, 7858, 2990, 3059, 3379, 3483, 3679, 4903, 5314, 6137, 6971, 6985, 7852, 9552, 7467, 3608, 3609, 3610, 9553, 3604, 3605, 3606, 8423, 3699, 3700, 3571, 19616, 19617, 9550, 9551]}, "n_points": 32, "state": "restart", "warm_start": false}`


Runner: Block finished: 32 paid measurements; 72 remain.

## Block 9 (after 178 of 250 measurements)

**Budget:** 178 used / 250 total; 72 remaining.
**Question:** Block 8 found nothing above 134 K. Which candidates still have a real chance of beating 137.4 K (19039)?
**Mode:** exploit, with a high-upside gamble.
**Gain:**
- The 2 H2S picks could raise the best a lot, to an estimated 150-200 K, if these entries are high-pressure measurements. If they come back low, they cost only 2 points.
- The other 23 points are tail draws from families measured at 130-134 K. Each has a small chance of 135-139 K.
**Evidence:** Block 8 (32 points) gave no improvement:
- Hg-1212 duplicates scored 113-134;
- Hg-1234 scored 116-127;
- Hg-Tl at wm 58-62 scored 76-127;
- Sr-Tl 122-128 and Hg-Sr 122-132;
- the unidentified 1223-like picks scored 50-71.

All Ca-Ba-Hg 1223 compositions are now measured (maximum 135.8).

New check (work/hyd.py): I decoded the 2-element candidates from mean and range of atomic mass. 20572 and 20573 decode as H-S, and their weighted mean mass of 11.36 matches H2S exactly. This is an inference from descriptors, not a measurement.

Domain knowledge (a prior, not a measurement): compressed hydrogen sulfide superconducts at about 150-203 K under high pressure. That is the only known material class above the Hg-cuprate record, so these two picks have by far the largest upside left. The other hydrides in the pool (Pd-H, Nb-H, Ta-H, SiH4) are low-Tc and skipped.

The other families picked have these measured values:
- Hg-Re duplicates of 8594 (133.1) and near-identical compositions (relatives at 130-133.1);
- Hg-1212 O-variants at wm 60.6-61.4, the neighbours of the 61.53 duplicates that reached 134;
- Hg-Tl at wm 65, the untested Tl members closest to 1212.
**Decision:** use fixed_points on 25 picks:
- H2S: 20572, 20573;
- Hg-Re: 8595, 9273, 9274, 10207-10209, 9334, 8593, 8592;
- Hg-1212 O-variants: 3280, 7776, 7777, 3427, 7775, 17972, 7774, 3281, 17971;
- Hg-Tl: 7478, 7466, 3165, 7479, 3607.

If H2S scores high, test any other hydrides next, and keep testing hydrides while that pays off. Otherwise use the remaining 47 points to finish the families with the best tails.
**Hyperparameters:** not applicable (fixed list).
**Block length:** 25. This covers the gamble plus the three best remaining families, and leaves 47 points to follow up whatever looks best.
**Review:** First, the H2S values. Then:
- whether any Re, Hg-1212 O-variant or Tl pick scored 135 K or more; if one did, finish that family (17965-17973, 7770-7773, 7778-7780);
- otherwise the remaining candidates are the Pb wm 53.8-54.8 remainder, the F remainder, Sr 7693/8421, Mo 3683, and kNN picks.

Proposal: `{"script": "lib/fixed_points.py", "label": "mix9", "args": {"seed": 9, "points": [20572, 20573, 8595, 9273, 9274, 10207, 10208, 10209, 9334, 8593, 8592, 3280, 7776, 7777, 3427, 7775, 17972, 7774, 3281, 17971, 7478, 7466, 3165, 7479, 3607]}, "n_points": 25, "state": "restart", "warm_start": false}`


Runner: Block finished: 25 paid measurements; 47 remain.

## Block 10 (after 203 of 250 measurements)

**Budget:** 203 used / 250 total; 47 remaining.
**Question:** H2S candidate 20573 scored 185 K. Can anything left in the pool beat it?
**Mode:** exploit (tail lottery). No exploration question would change the result.
**Gain:** Probably none. The best is very likely final at 185 K. This block takes the highest-ranked cuprate tail draws so the budget is still used.
**Evidence:**
- Block 9 measured the two H2S entries at 20572 = 60 K and 20573 = 185 K. The new best is 185 K, up from 137.4.
- The other 23 Block 9 points were 35-133.2 K, with no cuprate at 135 K or above.
- work/hyd4.py decoded every candidate with 2-5 elements whose lightest element is H. It uses the mean, range and geometric mean of atomic mass, so this is an inference, not a measurement. Results:
  - No other H-S compound and no H3S (weighted mean mass 8.77) is in the pool.
  - The other H-containing candidates are Pd-H, Nb-H, Ta-H, SiH4, Na-Co-O-H, H-C-alkali, Fe-As/Se hydrides and H-cuprates. All belong to classes with low Tc, at most about 90 K by domain prior.
- Domain prior (not a measurement): no known non-hydride superconductor gets near 185 K. Hg cuprates reach about 164 K at most under pressure.
**Decision:** use fixed_points on 24 picks, the leading unmeasured Hg-cuprate tail candidates:
- the Hg-Pb weighted-mean-mass 54 remainder (9);
- Hg-Re 5488, 6082 and 6083;
- Sr 7693 and 8421;
- Mo 3683;
- the 9 kNN top picks: 10354, 3673, 3674, 4523, 4524, 2387, 2388, 10355, 10356.

Each has only a small chance of reaching 135 K and essentially no chance of 185 K. I chose a fixed list because the library GP could not be smoke-tested in the sandbox (permission error), so a fixed list is the safe option.
**Hyperparameters:** not applicable.
**Block length:** 24. That leaves 23 for a final block, in case a surprise (a value above 150 K) shows up in a family worth finishing.
**Review:** Did anything exceed 150 K? If not, spend the last 23 on the remaining kNN or cuprate-tail picks (nn8.py list), or on diversity picks.

Proposal: `{"script": "lib/fixed_points.py", "label": "tail10", "args": {"seed": 10, "points": [3985, 3983, 3981, 6139, 4414, 6140, 10724, 5200, 4416, 5488, 6082, 6083, 7693, 8421, 3683, 10354, 3673, 3674, 4523, 4524, 2387, 2388, 10356, 10355]}, "n_points": 24, "state": "restart", "warm_start": false}`


Runner: Block finished: 24 paid measurements; 23 remain.

## Block 11 (after 227 of 250 measurements)

**Budget:** 227 used / 250 total; 23 remaining.
**Question:** None that could change the result. The best is 185 K (H2S, candidate 20573), and the domain prior says no other material class in the pool comes close to it.
**Mode:** exploit (final tail lottery).
**Gain:** Very likely none on the best value. This block spends the budget on the cuprate candidates most likely to score high.
**Evidence:**
- Block 10 scored 15.8 to 143 K. The highest were 143 K at the Hg-Pb weighted-mean-mass 54 picks 3985 and 5200; the kNN picks scored 110 to 118 K.
- None of the 24 scored above 150 K.
- Earlier hydride decoding found no other H-S compound in the pool (an inference from descriptors, not a measurement).
**Decision:** use fixed_points on 23 picks:
- the 10 nearest unmeasured neighbours (standardised distance) of the 143 K points 3985 and 5200: 7066, 7068, 3976, 8993, 9017, 9016, 3685, 5860, 5861, 5863;
- the top 13 by kNN-predicted Tc from nn8.py, each predicted around 126-131 K.
**Hyperparameters:** not applicable.
**Block length:** 23, the whole remaining budget. No later decision could use the answers.
**Review:** The campaign ends. The final best should be 185 K at 20573 unless a surprise appears.

Proposal: `{"script": "lib/fixed_points.py", "label": "final11", "args": {"seed": 11, "points": [7066, 7068, 3976, 8993, 9017, 9016, 3685, 5860, 5861, 5863, 2998, 8237, 8239, 8241, 6982, 2989, 4379, 3267, 3021, 2988, 6075, 10513, 4961]}, "n_points": 23, "state": "restart", "warm_start": false}`


Runner: Block finished: 23 paid measurements; 0 remain.
