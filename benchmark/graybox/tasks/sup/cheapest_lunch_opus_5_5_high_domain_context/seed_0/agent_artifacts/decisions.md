# Decisions

## Block 1 (after 0 of 250 measurements)

**Budget:** 0 used / 250 total; 250 remaining.
**Question:** Does the chemistry prior (Hg/Tl/Bi cuprates with ~1223 stoichiometry give the highest Tc) hold on this pool, and which cuprate family/stoichiometry region should the model-guided search focus on?
**Mode:** both (prior-guided exploitation + family comparison + 4 random baseline points).
**Gain:** Likely finds a >100 K material immediately if the prior holds; tells which family (Hg, Tl, Bi, Y) leads, which then restricts a GP/TuRBO search to that region. If cuprates do not lead, switch to a broad pool GP-BO with a space-filling design.
**Evidence:** No measurements yet. Pool of 21263 candidates, 81 composition descriptors (unweighted/weighted stats of element properties). Domain knowledge (prior, not measurement): highest Tc are Hg-Ba-Ca-Cu-O (~130+ K), Tl-Ba-Ca-Cu-O (~125 K), Bi(Pb)-Sr-Ca-Cu-O (~110 K), YBCO (~92 K). Unweighted mean/range of atomic mass identify element sets exactly (work/explore.py: 119 Hg-Ba-Ca-Cu-O, 123 Tl-Ba-Ca-Cu-O, 341 Bi-Sr-Ca-Cu-O, 720 Y-Ba-Cu-O candidates ...); weighted mean mass picks stoichiometry closest to ideal 1223/1212/2223 phases (work/prior.py). Many candidates have identical features (duplicate compositions).
**Decision:** experiment: fixed_points over prior picks: Hg1223 x3 (7788,7787,8358), Hg1212 x2 (17972,7775), Hg(Pb)1223 x2 (3981,3982), Tl2223 x2 identical-feature duplicates (981,1438; also measures label spread among duplicates), Tl1223 (17247), Bi2223 (678), BiPb2223 (478), Hg1201 (3624), Tl2201 (1340), YBCO (6131), random baseline (5736,10867,13542,18084, rng seed 0).
**Hyperparameters:** not applicable (fixed list).
**Block length:** 19 measurements: one-to-three per family is the smallest set that ranks families and gives a GP some contrast; ~8% of budget.
**Review:** rank families by Tc; spread between duplicates 981/1438; then run pool GP-BO (warm start) possibly restricted to the leading families' candidates (custom subset optimizer) or global with logEI.

Proposal: `{"script": "lib/fixed_points.py", "label": "prior_cuprates", "args": {"seed": 0, "points": [7788, 7787, 8358, 17972, 7775, 3981, 3982, 981, 1438, 17247, 678, 478, 3624, 1340, 6131, 5736, 10867, 13542, 18084]}, "n_points": 19, "state": "restart", "warm_start": false}`


Runner: Block finished: 19 paid measurements; 231 remain.

## Block 2 (after 19 of 250 measurements)

**Budget:** 19 used / 250 total; 231 remaining.
**Question:** Within the Hg-1223 neighbourhood (the leading family), which compositions/variants (pure Hg1223 O-series, the 21-row identical Hg1223 group, Re-, Tl-, Pb-substituted Hg1223, Hg1212) carry the highest Tc, and how large is the label spread among identical-feature rows?
**Mode:** both (exploit the best region found; compare sub-families and measure duplicate spread).
**Gain:** Expected to beat 135 K if any of these Hg1223 reports (e.g. pressure / fluorinated / Re-, Tl-doped studies) is higher; the result tells which sub-family to exhaust next, and whether the large duplicate groups (21 at wtd-mass 54.627, 16 at 61.534) are worth measuring fully.
**Evidence:** Block 1: Hg-Ba-Ca-Cu-O 1223 region best: 7787 = 135 K, 7788 = 134, 8358 = 118.5; HgPb1223 3982 = 130, 3981 = 125; Hg1212 7775 = 127, 17972 = 105.1; Tl2223 duplicates 981/1438 = 105 / 119.95 (identical features, 15 K spread -> features do not determine Tc; label noise among duplicates); Tl1223 112, Bi2223 110, BiPb2223 107, YBCO 93.4, Tl2201 85, Hg1201 35; random 98 (5736), 14, 3.8, 3.5. Domain prior (hypothesis): Hg1223 is highest-Tc family at ~133-135 K ambient, higher under pressure; F/Re/Tl substitution keeps ~130+ K. Element sets decoded from unweighted mean mass + range mass (work/hgfam.py, work/hgx.py).
**Decision:** experiment/exploit with fixed_points: 6 of the 21 identical Hg1223 rows (2804,3028,4035,5654,7105,8535); 10 unmeasured Hg1223 O-series rows (7789,7786,7124,3979,7784,7781,7783,8359,3426,7126); HgRe1223 5 (8594,9273,8596,8591,5487); HgTl1223 4 (4034,4031,4029,7468); HgPb1223 4 (3980,3983,4414,4416); Hg1212 3 (7776,2990,6137). If one sub-family/duplicate group stands out, exhaust it next; otherwise move to a subset-restricted GP over Hg/Tl cuprates.
**Hyperparameters:** not applicable (fixed list).
**Block length:** 32 measurements: 2-10 per sub-family is the smallest set to rank them and estimate duplicate spread; ~14% of remaining budget.
**Review:** best per sub-family; spread within 54.627 group; whether any point > 135 K; then choose: exhaust winning group, or subset GP-BO.

Proposal: `{"script": "lib/fixed_points.py", "label": "hg1223_scan", "args": {"seed": 0, "points": [2804, 3028, 4035, 5654, 7105, 8535, 7789, 7786, 7124, 3979, 7784, 7781, 7783, 8359, 3426, 7126, 8594, 9273, 8596, 8591, 5487, 4034, 4031, 4029, 7468, 3980, 3983, 4414, 4416, 7776, 2990, 6137]}, "n_points": 32, "state": "restart", "warm_start": false}`


Runner: Block finished: 32 paid measurements; 199 remain.

## Block 3 (after 51 of 250 measurements)

**Budget:** 51 used / 250 total; 199 remaining.
**Question:** Does any remaining Hg-1223-type candidate (rest of the 21-row identical group at wtd-mass 54.627, and the unmeasured Tl-, Re-, Pb-substituted Hg1223 rows, the x2 52.2-52.4 Hg rows, Hg1234) carry Tc above 135.7 K (e.g. high-pressure / optimally-doped reports)?
**Mode:** exploit (with a family-ranking side answer).
**Gain:** Features cannot discriminate identical-feature rows (Tc spread 107-135.7 K within the 54.627 group), so the only way to get the group's max is to measure it; the prior says Hg1223 is the record family (and pressure reports reach ~150-164 K). Expected: a few more 133-136 K values and a chance of a large jump. Answer also tells whether to move on to Hg1212 dup groups / a pool GP-BO next.
**Evidence:** Best 135.7 K (7105, 54.627 group); same group: 135, 134, 132.5, 131, 107. Hg1223 O-series 53.66-54.03: 132-135; 54.10-54.25: 94-110 (overdoped, skip). HgRe1223 130-133.1; HgTl1223 132-134 (7468=110 off-stoich); HgPb1223 118-130; Hg1212 105-128. Non-Hg families <=120 K. Domain prior (hypothesis): Hg1223 highest-Tc family.
**Decision:** fixed_points: 15 remaining 54.627-group rows; HgTl1223 3; HgRe1223 9; HgPb1223 9; Hg x2 52.2-52.4 singletons 6; Hg1234 group 4 (46 total). Skips overdoped O-series dups (7125, 7782, 7785).
**Hyperparameters:** not applicable (fixed list; indices from work/b3.py).
**Block length:** 46: exhausts the full high-prior Hg1223 neighbourhood in one pass; leaves 153 for model-guided search elsewhere.
**Review:** any value >135.7 and from which subgroup; then decide between Hg1212 61.534/60.x groups (pressure reports), Hg-F/other Hg families, or a warm-started pool GP-BO over the rest.

Proposal: `{"script": "lib/fixed_points.py", "label": "hg1223_exhaust", "args": {"seed": 0, "points": [2986, 2987, 3002, 3060, 3603, 4412, 4709, 4904, 5315, 5655, 5788, 5789, 6138, 7791, 19622, 4030, 4032, 4033, 5790, 8592, 8593, 8595, 9274, 9334, 10207, 10208, 10209, 3984, 3985, 4413, 4415, 4417, 5200, 6139, 6140, 10724, 5424, 5473, 5474, 5475, 5476, 5477, 3834, 4106, 5050, 7858]}, "n_points": 46, "state": "restart", "warm_start": false}`


Runner: Block finished: 46 paid measurements; 153 remain.

## Block 4 (after 97 of 250 measurements)

**Budget:** 97 used / 250 total; 153 remaining.
**Question:** With the Hg1223 neighbourhood mostly used up (best 143 K from HgPb1223), do the other Hg cuprates (the 16-row identical Hg1212 group at wtd-mass 61.534, Hg1212 singletons near the 127 K point, and Hg1223/Hg1245 with other substituents: Pb, Tl, F, Nd, Bi, Sr, Sb, Zr, Mo, Cr, V) include any rows above 143 K?
**Mode:** exploit (the next most likely high-Tc rows), which also ranks the remaining Hg families.
**Gain:** Duplicate groups hold a wide spread of labels (HgPb rows with identical features gave 130 vs 143; Hg1223 group 107-135.7), so large groups can contain high-pressure reports. The prior says Hg1212 under pressure reaches about 150 K and fluorinated Hg1223 about 160 K. Expect a chance at >143 K. If nothing reaches 143, I switch next to a warm-started pool GP-BO / nearest-neighbour exploitation or to Tl families.
**Evidence:** Block 3: two HgPb1223 rows were 143 K (3985 at x2 53.78; 5200 at 54.75, which has the same features as 4415 = 130). Remaining 54.627 Hg1223 rows were 110-135.5. HgRe 130-133.2, HgTl 132-134, Hg x2 52.2-52.4 132.9-135.8, Hg1234 116-127. Hg1212 so far: 128, 127, 122, 116, 105. Element sets were decoded from unweighted mean and range of mass (work/hgx2.py). work/nn.py ranks unmeasured rows by standardized distance to the measured rows of 133 K or more; its top hits are these same groups. Domain prior (a hypothesis, not data): Hg cuprates hold the Tc record.
**Decision:** fixed_points: 14 unmeasured Hg1212 61.534 dup rows; 5 Hg1212 singletons at x2 60.6-60.9; 22 substituted Hg-Ba-Ca-Cu-O rows (Pb 56.44 x2, Pb 47.21, Tl 52.53/51.17, F 52.53, Nd, Bi, Sr x2, Sb x2, Zr, Mo x2, Cr x2, V x2, Hg1245 x2, X~105 row). Skip the overdoped Hg1223 dup rows (siblings were 94-110 K).
**Hyperparameters:** not applicable (fixed list).
**Block length:** 41 measurements. This covers the remaining high-prior Hg rows and leaves 112 for model-guided search.
**Review:** Did anything beat 143 K, and in which family? Then choose between the rest of the Hg1212 singletons, Hg-F / Hg1201 groups, and a warm-started GP-BO (or kNN) over the pool.

Proposal: `{"script": "lib/fixed_points.py", "label": "hg_variants", "args": {"seed": 0, "points": [2991, 3059, 3379, 3483, 3669, 3678, 3679, 4903, 5314, 6971, 6985, 7852, 8037, 9552, 3427, 7774, 3280, 7777, 7773, 7066, 7068, 19039, 7469, 5318, 8249, 3693, 3698, 10720, 6130, 10100, 3682, 3683, 3686, 3687, 3680, 3681, 3976, 4080, 4081, 9545, 3699]}, "n_points": 41, "state": "restart", "warm_start": false}`


Runner: Block finished: 41 paid measurements; 112 remain.

## Block 5 (after 138 of 250 measurements)

**Budget:** 138 used / 250 total; 112 remaining.
**Question:** Block 4 found nothing above 143 K. Do the rest of the high-prior rows hold anything above 143 K? These are the Hg1212 singletons, Hg1212/Hg1223 variants with Tl, Pb, Re, F, Sr, Mo or Bi substituted, the overdoped Hg1223 duplicate partners, and the 12 unmeasured rows in the duplicate Tl2223 group.
**Mode:** exploit (the next most likely high-Tc rows), which also ranks the remaining families.
**Gain:** Duplicate and near-duplicate rows have a wide label spread (130 vs 143 for HgPb rows with identical features; Hg1212 61.53 group 92-134), so only measuring finds high reports, such as high-pressure ones. Expect mostly 110-135 K, with a small chance of more than 143. The result decides whether the last ~72 go to the Tl/Hg1201 duplicate groups or to a model-guided pool search.
**Evidence:** Block 4 (41 pts): best 137.4 (19039, HgTl1223 at x2 52.53). The other values were HgF1223 5318 = 135, Hg1212 group 92-134, Hg1212 singletons 121-127, and substituted Hg1223 74-134. Overall best is still 143 (3985, 5200 HgPb1223). Element sets were decoded from the mean and range of atomic mass (work/hgx2.py, work/fam.py). The standardized nearest-neighbour ranking to rows of 133 K or more (work/nn.py) puts these same rows first. Domain prior (a hypothesis): Hg cuprates hold the Tc record, Hg1212 reaches ~150 K under pressure, and Tl2223 reaches ~125 K.
**Decision:** fixed_points on 40 rows:
- Hg1212 singletons: 10 rows
- substituted Hg1212 rows (Sr, Mo, Tl, F, Re): 10 rows
- HgPb at x2 64.95: 3 rows
- HgTl at x2 ~65: 3 rows
- other rows: HgSr 3700, Hg-rich Ca-rich 4082, (Hg,Tl)(Ba,Sr) 3604/3606, Tl-?-Ca 18608, (Bi,Hg)SrCaCuO 5860/5861
- Tl2223 duplicate group: 5 rows
- overdoped Hg1223 duplicate partners 7125/7785: 2 rows
**Hyperparameters:** not applicable (fixed list).
**Block length:** 40 measurements. This covers the high-prior Hg rows that are left and leaves 72 for the final phase.
**Review:** Did anything beat 143 K? If a subgroup is high, exhaust its relatives. Otherwise spend the rest on the remaining Tl2223/Tl1223 duplicate groups and Hg1201 groups, or on a warm-started pool GP.

Proposal: `{"script": "lib/fixed_points.py", "label": "hg1212_subst", "args": {"seed": 0, "points": [3281, 7769, 7770, 7771, 7772, 7778, 7779, 7780, 17973, 17971, 3571, 3672, 3608, 3609, 3610, 7467, 9553, 5317, 5488, 8422, 8993, 9016, 9017, 7466, 3165, 3607, 3700, 4082, 3604, 3606, 18608, 5860, 5861, 995, 1000, 1480, 1531, 2045, 7125, 7785]}, "n_points": 40, "state": "restart", "warm_start": false}`


Runner: Block finished: 40 paid measurements; 72 remain.

## Block 6 (after 178 of 250 measurements)

**Budget:** 178 used / 250 total; 72 remaining.
**Question:** Can any row beat 143 K now that the Hg-1223/1212 neighbourhood is nearly used up? Two candidates: (a) the H2S binary rows. A domain prior (hypothesis): compressed H2S/H3S was reported at about 190-203 K. (b) The unmeasured Hg-containing rows near Hg1223/Hg1212 stoichiometry: 7-element variants, V/Sr/W/Cr substituted rows, and rows at x2 57.3.
**Mode:** exploit (both are top-prior rows), with a side answer on whether any family beyond the Hg cuprates holds the max.
**Gain:** If the H2S rows carry a high-pressure label (prior about 200 K), the best jumps by about 50 K. The Hg rows are a cheap check of the last cuprate rows that could plausibly exceed 143 K, given the label spread within duplicate groups.
**Evidence:** Block 5 (40 pts) gave nothing above 143. Its best was 132 (3571), and Tl2223 dups were 110-129. The overall best is still 143 K (3985, 5200, HgPb1223); next are 137.4, 135.8 and 135.7. work/light.py decodes binaries from mean and range of atomic mass. It finds H-S binaries 20572 and 20573 with wtd mean mass 11.36, which means H2S exactly. It also finds SiH4 (17816), which I skip because its prior Tc is low. work/hgall.py lists the 115 unmeasured rows whose mass range equals Hg-O. Those near Hg1223/1212 stoichiometry (x2 50-62) are chosen here: 7-element 4961-4964, 3471/3472, 21078-21080, 9335-9337 ((Hg,Re)(Ba,Sr)CaCuO-like), 5309, 8236; V-substituted 3639/3640/3670/3671 (siblings 120/127); Sr-substituted 7693/8421 (siblings 126-133.5); W 3673/3674; 3774, 8237/8239, 4773, 10107/10108; plus one row from each of the three x2 57.3 groups (9257, 9264, 9252).
**Decision:** fixed_points, 33 rows (new label h2s_hgrest). Hypothesis: H2S carries a pressure Tc far above the cuprates. If it does, the best becomes about 200 K and the rest of the budget goes to remaining sibling or high-prior rows. If not, the hydride prior fails for this pool, and the last 39 go to the remaining Hg-O-range rows and the Tl dup groups.
**Hyperparameters:** not applicable (fixed list).
**Block length:** 33 measurements. That covers the highest-prior rows that are left and keeps 39 for a final decision.
**Review:** H2S values. Are any Hg rows above 137? If yes, exhaust their duplicate group or series (e.g. the other 9257-9268/9252-9256 rows).

Proposal: `{"script": "lib/fixed_points.py", "label": "h2s_hgrest", "args": {"seed": 0, "points": [20572, 20573, 4961, 4962, 4963, 4964, 3471, 3472, 21078, 21079, 21080, 9335, 9336, 9337, 5309, 8236, 3639, 3640, 3670, 3671, 7693, 8421, 3774, 8237, 8239, 3673, 3674, 4773, 10107, 10108, 9257, 9264, 9252]}, "n_points": 33, "state": "restart", "warm_start": false}`


Runner: Block finished: 33 paid measurements; 39 remain.

## Block 7 (after 211 of 250 measurements)

**Budget:** 211 used / 250 total; 39 remaining.
**Question:** H2S gave 185 K (20573), a new best that is 42 K above the cuprates. Is there any other row that could beat it?
**Mode:** exploit (the last high-prior rows).
**Gain:** Probably none. 185 K is very likely the maximum of the pool. A full scan of binary hydrides (binaries whose lighter element decodes to H, an inline script run this call) finds only SiH4 (17816, prior about 17 K), H-S (both rows measured: 60 and 185), and Nb-H, Pd-H and Ta-H (prior below 15 K). No other H-S row, so no H3S row, exists. Multi-element rows that contain H are hydrogenated YBCO-type cuprates (density range = Cu − H), which have a prior near 90 K. The budget must still be spent. The rows with the most upside left are the unmeasured Hg-O cuprate rows, where duplicate-label spread has produced outliers before.
**Evidence:** Block 6: H2S 20572 = 60 K and 20573 = 185 K. The Hg variants in that block were 51-131 K, so none reached 137. Before that the best was 143 K (HgPb1223). Domain prior: compressed H2S/H3S is reported at about 185-203 K, which matches the measurement. No other family in the pool has a prior near that level.
**Decision:** fixed_points over 20 leftover Hg-O-range rows: overdoped Hg1223 dup 7782; the x2 57.3 groups 9258, 9253, 9265; the 5-element x1 92.74 Hg series 3494-3500, 8911, 8912, 9202; and 4996, 4997, 5054, 8235, 17966, 2988.
**Hyperparameters:** not applicable (fixed list).
**Block length:** 20 measurements. This leaves 19 for a final block, in case a surprise turns up (e.g. a row above 143 whose group should be exhausted).
**Review:** Did anything exceed 143 (or 185)? If not, spend the last 19 on the remaining leftover rows (Tl2223/Tl1223 dup groups, rest of the x2 57.3 groups, 5055-5060, 8240-8243), or on a warm GP/kNN pick.

Proposal: `{"script": "lib/fixed_points.py", "label": "final_hg", "args": {"seed": 0, "points": [7782, 9258, 9253, 9265, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 8911, 8912, 9202, 4996, 4997, 5054, 8235, 17966, 2988]}, "n_points": 20, "state": "restart", "warm_start": false}`


Runner: Block finished: 20 paid measurements; 19 remain.

## Block 8 (after 231 of 250 measurements)

**Budget:** 231 used / 250 total; 19 remaining.
**Question:** Can any row that is left beat 185 K (H2S)? Almost certainly not. The open question is only which leftover rows have the highest upside.
**Mode:** exploit (final block, uses the whole remaining budget).
**Gain:** The chance of beating 185 K is very small. The binary-hydride scan found no other high-prior hydride. These rows are the leftovers with the highest prior, where label spread among duplicates could still produce outliers.
**Evidence:** Block 7 (20 Hg-O leftover rows) gave 9-110 K. The best were 2988 = 110 and 17966 = 105.1, and nothing reached 143. Best overall: 185 K (20573, H2S); next 143 (HgPb1223). Domain prior: Tl2223 reaches ~125 K and Tl1223 ~110-120 K. Siblings of 2988 and 17966 are 2989 and 17967-17970.
**Decision:** fixed_points over 19 unmeasured rows:
- Tl2223 duplicate group at x2 58.64: 2075, 2747, 3088, 4100, 4101, 6135, 8110
- siblings of the 105-110 K rows: 17967-17970, 2989
- Tl1223 group at x2 52.58: 3899, 4097-4099, 5323, 6092, 17007
**Hyperparameters:** not applicable (fixed list).
**Block length:** 19, which is the whole remaining budget.
**Review:** Campaign ends. The final best is max(185, block 8 values).

Proposal: `{"script": "lib/fixed_points.py", "label": "final_tl", "args": {"seed": 0, "points": [2075, 2747, 3088, 4100, 4101, 6135, 8110, 17967, 17968, 17969, 17970, 2989, 3899, 4097, 4098, 4099, 5323, 6092, 17007]}, "n_points": 19, "state": "restart", "warm_start": false}`


Runner: Block finished: 19 paid measurements; 0 remain.
