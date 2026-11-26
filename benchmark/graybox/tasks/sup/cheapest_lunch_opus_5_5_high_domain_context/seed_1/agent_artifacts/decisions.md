# Decisions

## Block 1 (after 0 of 250 measurements)

**Budget:** 0 used / 250 total; 250 remaining.
**Question:** Does the domain prior (cuprates, especially Hg- and Tl-based multi-layer cuprates, have the highest Tc) hold in this pool, and which cuprate family/sub-region is best to exploit?
**Mode:** both (prior-guided exploitation + family-level exploration).
**Gain:** Likely finds a high-Tc cuprate immediately (prior: Hg-1223 ~130+ K, Tl-2223 ~125 K, Bi-2223 ~110 K, YBCO ~92 K); ranks families so the next block (GP-BO restricted to the best family subset) searches the right region.
**Evidence:** No measurements yet. Pool of 21263 candidates (6093 exact duplicate feature rows), 81 composition-derived descriptors. Domain knowledge used (prior, not measurement): cuprates are the high-Tc class. From public features: range_ThermalConductivity (x67)=399.97 identifies Cu+O compounds (10419 candidates); range_atomic_mass (x7) then identifies the heaviest element: 184.59=Hg (571), 188.38=Tl (613), 192.98=Bi (1687), 121.33=Ba-max (YBCO-type, 2059), 122.91=La-max (1567). Hg-Ba-Ca-Cu-O has unweighted mean mass x1=91.51; wtd_mean mass x2≈54.6 matches HgBa2Ca2Cu3O8 (Hg-1223), ≈61.5 matches Hg-1212; x2≈58.64 in Tl-Ba-Ca-Cu-O matches Tl-2223. Assumption: these stoichiometric identifications are approximate.
**Decision:** experiment: fixed design of 26 candidates: 5 Hg-1223-like (distinct x2 53.7-54.6), 3 Hg-1212-like, 5 other Hg cuprates, 4 Tl 5-element (incl. Tl-2223), 3 other Tl, 3 Bi (>=5 elements), 2 Y/Ba-type, 1 La-type; within groups chosen by maximin in standardized features. If Hg/Tl dominate (>120 K) next block exploits that subset with a GP; if not, the prior is wrong and I widen to a GP over the full pool.
**Hyperparameters:** not applicable (fixed points; list in work/design1.json, built by work/design1.py + design1b.py).
**Block length:** 26 measurements (~10% of budget): enough to rank 5 families with replication within the top ones before committing to a model.
**Review:** Tc by family; whether Hg-1223 duplicates/near-duplicates vary widely (label noise among identical features); fit a GP on cuprate subset and check LOO prediction.

Proposal: `{"script": "lib/fixed_points.py", "label": "prior_design", "args": {"seed": 0, "points": [7783, 3426, 2804, 8358, 3979, 7774, 17973, 17970, 8862, 9958, 17969, 3897, 9964, 19644, 5539, 1693, 1014, 3740, 2696, 1561, 6066, 4951, 7648, 4795, 17622, 981]}, "n_points": 26, "state": "restart", "warm_start": false}`


Runner: Block finished: 26 paid measurements; 224 remain.

## Block 2 (after 26 of 250 measurements)

**Budget:** 26 used / 250 total; 224 remaining.
**Question:** Do the duplicate records of the best composition (Hg-1223, 21 identical feature rows) and of Hg-1212 (16 rows) carry higher Tc labels than the single ones measured (e.g. pressure-enhanced records)?
**Mode:** exploit (with a test of the "duplicate rows carry different labels" hypothesis).
**Gain:** Domain prior: HgBa2Ca2Cu3O8+d is the ambient record holder (~133-135 K) and reaches ~150-164 K under pressure; Hg-1212 reaches ~150 K under pressure. Composition-only features cannot separate these records, so the only way to find them is to measure the duplicates. Expected: several values 130-160 K; best likely improves from 133 K. Also tells how wide the label spread within one feature row is, which decides whether the next model should treat duplicates as noise.
**Evidence:** Block 1 confirmed the prior: Hg cuprates best (133.0 K idx 3426, 132.5 K idx 2804 = Hg-1223 composition, 130.1 K idx 3897 = 6-element Hg-1223 with F, 122 K Hg-1212); Tl-cuprates 105-111 K; Bi 13-80 K; YBCO-type up to 116 K; La-type 23 K. Hg-1212 singles 105-122 K. Group 2804 has 20 unmeasured identical rows; group 2990 (Hg-1212, x2=61.53) has 16.
**Decision:** experiment/exploit: fixed_points over all 20 remaining Hg-1223 duplicates (group of 2804) plus 10 random of the 16 Hg-1212 duplicates (group of 2990). If spread is large and max >140 K, next block samples other large duplicate groups of high-Tc compositions (Hg-1201 group 2803 x28, Tl groups) and the rest of 2990; if spread is small, move to GP-BO over the Hg/Tl cuprate subset of distinct compositions.
**Hyperparameters:** not applicable (list in work/design2.json, built by work/design2.py, seed 2 for the Hg-1212 subsample).
**Block length:** 30: the duplicate rows are exchangeable to any model, so there is no in-block learning to wait for; all 20 Hg-1223 rows guarantees the group max.
**Review:** distribution of labels within each group; max; whether duplicates' spread justifies sampling other big duplicate groups (Hg-1201 2803, Tl 990/981, Hg 3373/5321).

Proposal: `{"script": "lib/fixed_points.py", "label": "dup_exploit", "args": {"seed": 0, "points": [2986, 2987, 3002, 3028, 3060, 3603, 4035, 4412, 4709, 4904, 5315, 5654, 5655, 5788, 5789, 6138, 7105, 7791, 8535, 19622, 6137, 3679, 3678, 8037, 3059, 2990, 7852, 9552, 6985, 5314]}, "n_points": 30, "state": "restart", "warm_start": false}`


Runner: Block finished: 30 paid measurements; 194 remain.

## Block 3 (after 56 of 250 measurements)

**Budget:** 56 used / 250 total; 194 remaining.
**Question:** Which remaining Hg-cuprate records carry Tc above the 135.7 K incumbent? In particular, do the Hg-1223 rows whose oxygen was apparently recorded as "O1" (likely O8+δ notation: group of 9 identical rows), the F-doped Hg-1223 series, the Tl-substituted Hg-1223 series, and the last Hg-1212 duplicates do so?
**Mode:** exploit, with a test of a structural hypothesis.
**Gain:** Block 2 found the Hg-1223 duplicate group spans 107-135.7 K (mean ~127); best rose from 133.0 to 135.7 K (idx 7105). The Hg-1212 group spans 105-134 K over 10 rows. Each still-unmeasured record of an equivalent compound is another draw from a similar wide spread, and domain knowledge says (Hg,Tl)-1223 and fluorinated Hg-1223 reach about 138 K, and pressure records reach about 150-164 K. Expected: some chance (maybe 30-50%) of beating 135.7 K. The answer also decides whether the next block should go to the other "O1"-notation Hg groups (Hg-1234 x6, Hg-1245 x9, Hg-1201) or to Hg-Pb/Hg-Re-1223 and the oxygen-series singles.
**Evidence:** I recovered stoichiometries offline from mass descriptors (wtd mean, gmean, entropy, std; script work/decode.py). This is exact for the 5-element Hg-Ba-Ca-Cu-O set up to non-uniqueness. Confirmed fits: x2=54.63 is HgBa2Ca2Cu3O8. x2=84.67 (group of 3373, 9 rows) matches HgBa2Ca2Cu3O1 exactly (mass 762.04/9). That is the same compound with the oxygen written as a placeholder, which is how δ-notation and pressure records are often stored. x2=94.06 (3380) is Hg-1212 "O1", 78.70 is Hg-1234 "O1", and 74.56 is Hg-1245 "O1". 3897 (F-doped Hg-1223 with low-O notation, x2 80.57) measured 130.1 K, and 8 sibling rows are unmeasured. (Hg,Tl)-1223 rows 4029-4034 have x2 54.64-54.84, nearly identical to Hg-1223. Hg-1223 off-stoichiometric O singles measured 101-133 K, so they are lower priority. Domain prior used: Hg-1223 holds the ambient record of about 134-138 K (Tl/F variants) and is highest under pressure.
**Decision:** exploit/experiment with fixed_points over 32 candidates: all 9 Hg-1223 "O1" rows; the 6 remaining Hg-1212 duplicates plus Hg-1212 "O1" (3380); 9 F-doped Hg-1223 siblings of 3897; 6 (Hg,Tl)-1223 rows (4029-4034) plus 5855 ((Hg,Tl)-1223 "O1"-like). If the "O1" group shows values above 136 K, next I sample the other "O1" Hg groups (Hg-1234 4419 x6, Hg-1245 5321 x9, Hg-1201 O1). If not, I go to Hg-Pb/Hg-Re 1223 and the oxygen series, and then to a pool GP over the cuprate subset for the rest.
**Hyperparameters:** not applicable (fixed list in work/design3.json, built by work/design3.py).
**Block length:** 32. These are exchangeable records whose selection does not depend on each other's outcomes, so in-block adaptation gains nothing. 32 covers the four highest-prior subgroups completely and leaves 162 for follow-up.
**Review:** max and spread per subgroup; whether the "O1" notation rows are high-Tc (pressure/δ records); choose the next Hg subgroups accordingly.

Proposal: `{"script": "lib/fixed_points.py", "label": "hg_variants", "args": {"seed": 0, "points": [3373, 3381, 3891, 4551, 4579, 4580, 5339, 5340, 7642, 2991, 3379, 3483, 3669, 4903, 6971, 3380, 3898, 3927, 3896, 3895, 3894, 3893, 5342, 3892, 5341, 4029, 4030, 4031, 4032, 4033, 4034, 5855]}, "n_points": 32, "state": "restart", "warm_start": false}`


Runner: Block finished: 32 paid measurements; 162 remain.

## Block 4 (after 88 of 250 measurements)

**Budget:** 88 used / 250 total; 162 remaining.
**Question:** Can anything in the pool beat the Hg-1223 ceiling of about 135 K? First, the only high-pressure hydride in the pool (H2S). Second, the untested Hg-1223 oxygen-series records and the Pb-doped Hg-1223 records.
**Mode:** exploit, plus a cheap high-upside test of a domain prior.
**Gain:** H2S under pressure is reported at 190-203 K (Drozdov 2014/2015, the conventional-superconductivity record). If the pool's two H2S records carry those values, the best rises by about 55-65 K with only 2 measurements. The other 27 are the next most likely Hg-1223-type records to reach 136-138 K. That gain is small but has non-trivial probability.
**Evidence:** Block 3 (32 pts) gave no improvement. The Hg-1223 "O1" group reached 123-135 K, (Hg,Tl)-1223 132-134 K, F-Hg-1223 124-131 K and Hg-1212 92-127 K. Every Hg-1223-type record clusters at 130-135.7 K, so ambient Hg cuprates look saturated near 135.7 K (incumbent idx 7105). Offline search of the public features found the 2-element hydrides: rows 20572 and 20573 have x0=2, mean mass 16.536 and range 31.057, which is exactly H+S. Their weighted mean mass of 11.36 gives an H fraction of 0.667, so the compound is H2S. The others are SiH4 (17816), PdH, NbH and TaH, all low-Tc by domain knowledge. The Hg-1223 oxygen series (O8.17-O8.29, plus the off-stoichiometric 9.7-9.9 decodes) and Hg-Pb-1223 (x2 53.8-54.8) are unmeasured. Measured siblings ranged 101-133 K. Domain prior used: H2S/H3S at pressure has the highest Tc. Pb- and Re-doped Hg-1223 reach about 130-135 K.
**Decision:** fixed_points over 29 candidates: the 2 H2S records, 12 Hg-1223 oxygen-series singles and 15 Hg-Pb-1223 records. If H2S is about 190 K or higher, the campaign best is likely final. The remaining budget then goes to a cheap sweep of other plausible pressure records plus model-guided exploitation. If H2S is low, the hydride prior fails for this pool. I then continue with Hg-Re-1223, the Hg-1234 "O1" group and (Hg,Tl) records, followed by a pool GP over the cuprates.
**Hyperparameters:** not applicable (fixed list in work/design4.json, built by work/design4.py).
**Block length:** 29. The records are exchangeable and chosen by prior, so measuring them in sequence teaches nothing within the block. This covers the top remaining hypotheses and leaves 133.
**Review:** the H2S values; the max over the oxygen-series and Pb records; whether any Hg-1223-type record exceeds 135.7.

Proposal: `{"script": "lib/fixed_points.py", "label": "h2s_hg", "args": {"seed": 0, "points": [20572, 20573, 8359, 7788, 7789, 7787, 7124, 7786, 7125, 7784, 7785, 7126, 7781, 7782, 3984, 3985, 3983, 3982, 3981, 3980, 4413, 6139, 4414, 6140, 10724, 4415, 5200, 4416, 4417]}, "n_points": 29, "state": "restart", "warm_start": false}`


Runner: Block finished: 29 paid measurements; 133 remain.

## Block 5 (after 117 of 250 measurements)

**Budget:** 117 used / 250 total; 133 remaining.
**Question:** H2S has given a best of 185 K. Is there any other record in the pool that could match it, and which records are the highest-Tc ones still unmeasured?
**Mode:** exploit (prior-ranked records), with a small check of the last remaining pressure hydride.
**Gain:** Block 4 raised the best from 135.7 to 185 K. The two H2S records gave 60 K and 185 K (idx 20573). The domain prior says nothing else composition-based is likely to beat about 165 K (the Hg-1223 pressure record), so the chance of beating 185 K is small. Still, these records have the highest Tc probability of anything left: Hg-Pb-1223 gave 143 K twice (idx 3985 and 5200). That shows pressure or record-high Hg-1223-family labels exist among the unmeasured siblings.
**Evidence:** Block 4 results: H2S 185 and 60 K. Hg-1223 oxygen-series singles 94-135 K. Hg-Pb-1223 (x0=6, x1=110.79) 103-143 K. I searched offline for hydrides: 2-element (x1 - x7/2 = 1.008) and 3- or 4-element (assumed min mass 1.008, checked against mean, range, gmean and std; work/hyd3.py). The only H-S compounds are the two H2S records, so no H3S is present. The other H-containing candidates look like Pd-H, alkali-intercalated hydrocarbon or organic-type compounds (low prior), plus SiH4 (17816; about 17 K at pressure in the literature). Domain prior used: Hg-Re- and Hg-Pb-1223 and Hg-1234 are high-Tc (130-135 K ambient, more under pressure), and H2S/H3S hold the composition record.
**Decision:** exploit with fixed_points over 38 prior-ranked records. These are 14 Hg-Re-1223 (x2 54.4-54.6), 6 Hg-1234 "O1", 10 Hg-1223/1234 lower-x2 variants (50.48 group, 52.2-52.4 series), 5 (Hg,Tl) records (3756-3760), 2 remaining Hg-Pb siblings (7066, 7068) and SiH4. If anything reaches 150 K or more, next I sample its siblings. Otherwise the final ~95 measurements go to the remaining Hg/Tl groups (Hg-1245 O1, Hg-1201) and a pool GP for any unexpected high-Tc family.
**Hyperparameters:** not applicable (fixed list in work/design5.json, built by work/design5.py).
**Block length:** 38. The records are exchangeable within their prior ranking and none of the values would change which other ones I pick inside this block. This leaves 95 to react.
**Review:** max per subgroup (Hg-Re, Hg-1234, Hg-Tl), whether any value ≥150 K appears, and SiH4's value. Then decide between continuing the prior list and a pool GP (gp_bo, warm start).

Proposal: `{"script": "lib/fixed_points.py", "label": "hg_re_more", "args": {"seed": 0, "points": [8596, 5790, 8595, 8594, 9273, 9274, 10207, 10208, 10209, 8593, 9334, 8592, 8591, 5487, 4419, 4710, 5319, 5433, 5434, 5435, 3834, 4106, 5050, 7858, 5473, 5474, 5475, 5476, 5477, 5424, 3756, 3757, 3758, 3759, 3760, 17816, 7066, 7068]}, "n_points": 38, "state": "restart", "warm_start": false}`


Runner: Block finished: 38 paid measurements; 95 remain.

## Block 6 (after 155 of 250 measurements)

**Budget:** 155 used / 250 total; 95 remaining.
**Question:** Do any of the remaining unmeasured Hg- or (Hg,Tl)-cuprate records (the highest-prior records left) reach the 150-165 K pressure-record range, or come anywhere near the 185 K H2S incumbent?
**Mode:** exploit (prior-ranked records).
**Gain:** The chance of beating 185 K is small. Domain knowledge puts the composition-based ceiling at about 165 K (Hg-1223 under pressure), and 185 K may well be the pool maximum. These records are still the most likely to hold high values, and spending the budget is required. The block also shows whether any unexpected group is high; if one is, its siblings get the next block.
**Evidence:** Block 5 (38 points) gave no gain. Hg-Re-1223 ranged 130-133.2 K, Hg-1234 "O1" 95-125 K, the lower-x2 Hg series 116-135.8 K and (Hg,Tl) 107.5-122.5 K. The remaining Hg-Pb siblings gave 131.5 and 133 K, and SiH4 gave 17 K. Across 150+ cuprate records the ceiling stays near 136 K, with two Hg-Pb records at 143 K. The best remains 185 K (H2S, idx 20573).
**Decision:** switch to a new fixed_points list: all 51 remaining Hg/(Hg,Tl) records identified offline from the public features. That is 9 Hg-1245 "O1" (x2=74.56), 16 Hg-1212-series singles (x2 60.4-61.4), 6 Hg-Re with higher x2, 6 Hg-Pb leftovers, 7 (Hg,Tl) 6-element records, and 7 Hg O-series singles (4711, 17963-17968). If any reaches 145 K or more, its siblings come next. Otherwise the final ~44 measurements go to a warm-started pool GP (gp_bo) to probe unexpected families.
**Hyperparameters:** not applicable (list in work/design6.json, built by work/design6.py).
**Block length:** 51. The records are exchangeable within the prior ranking, so in-block adaptation gains nothing. 51 exhausts the Hg family list and leaves 44 for a model-guided final block.
**Review:** max per subgroup; any value of 145 K or more; then set up a gp_bo warm-start block for the remaining 44.

Proposal: `{"script": "lib/fixed_points.py", "label": "hg_rest", "args": {"seed": 0, "points": [5321, 5429, 5430, 5431, 5432, 17216, 17222, 17574, 17575, 7780, 7779, 7778, 3280, 7776, 7777, 3427, 7775, 17972, 7773, 7772, 7771, 7769, 7770, 3281, 17971, 5488, 6083, 6082, 5785, 5786, 5787, 8993, 9016, 9017, 8475, 7092, 3976, 4812, 4813, 5852, 5853, 5854, 6202, 6204, 4711, 17963, 17964, 17965, 17966, 17967, 17968]}, "n_points": 51, "state": "restart", "warm_start": false}`


Runner: Block finished: 51 paid measurements; 44 remain.

## Block 7 (after 206 of 250 measurements)

**Budget:** 206 used / 250 total; 44 remaining.
**Question:** After hand-picking by domain prior, is there any family of records we have not identified that sits above the cuprate ceiling (about 143 K) or near the 185 K incumbent?
**Mode:** exploit with exploration (model-guided search over the pool).
**Gain:** Probably small. By domain prior, 185 K (H2S, idx 20573) is very likely the pool maximum. A warm-started GP over all 21k unused candidates is the cheapest way to probe families that the prior-driven lists skipped. It spends the budget on the model's best guesses rather than on known ~100 K groups.
**Evidence:** Block 6 (51 records, every remaining Hg/(Hg,Tl) record) gave no gain: maximum 135 K (5785), and most were 85-127 K. A few Hg-Tl 6-element records came in at 35-60 K. The ceiling across about 200 cuprate records stays at 136 K, apart from two Hg-Pb records at 143 K. The best remains 185 K. Hg-1201 (prior about 97 K) cannot beat it. Domain prior used: no composition-based superconductor exceeds about 165 K.
**Decision:** switch to lib/gp_bo with warm_start, using all 206 measurements (Matern-ARD GP, LogEI, scored over all unused pool candidates). If it finds a new high group (145 K or more), the final 22 measurements go to that group's siblings. Otherwise I continue gp_bo or move to UCB for the remainder.
**Hyperparameters:** gp_bo library defaults (logei, pool_chunk_size 512). The warm start already exceeds n_init, so no random initial design is paid for. Seed 7.
**Block length:** 22, half of what remains, so the final block can react if the GP finds something.
**Review:** values of the GP picks and which families they come from (decode with work/decode.py), and any value of 145 K or more.

Proposal: `{"script": "lib/gp_bo.py", "label": "gp_final", "args": {"seed": 7, "acquisition": "logei"}, "n_points": 22, "state": "restart", "warm_start": true}`

## Note after 206 measurements

The optimizer of block 7 (gp_final) crashed; its state was not saved, so a later block for this label must restart.

```
PermissionError: [Errno 13] Permission denied: '/proc/self/maps'
```


Runner: Block finished: 0 paid measurements; 44 remain.

## Block 8 (after 206 of 250 measurements)

**Budget:** 206 used / 250 total; 44 remaining.
**Question:** Block 7 (gp_bo) crashed on its first ask in this sandbox (PermissionError on /proc/self/maps, apparently from the torch/GP runtime) and measured nothing. Which records should the remaining budget go to, using an optimizer that runs here?
**Mode:** exploit (prior-ranked records).
**Gain:** Probably none against the 185 K incumbent (H2S, idx 20573). By domain prior, no composition-based record exceeds about 165 K, so 185 K is very likely the pool maximum. The budget must still be spent, and Tl-Ba-Ca-Cu-O are the highest-prior unmeasured family: Tl-2223 reaches about 125-128 K in the literature, and Tl-1223/1234 about 110-120 K. Duplicate feature rows carry different labels, so sampling several duplicates of the best stoichiometry could raise its observed maximum.
**Evidence:** The Tl-Ba-Ca-Cu-O group (x0=5, x1=92.27, x7=188.38, Cu+O flag x67=399.97) has 123 records but only 4 measured: 111.5, 108, 105 and 84 K. I identified stoichiometries by mean atomic mass x2: 58.64 is Tl2Ba2Ca2Cu3O10 (14 duplicate rows, one measured at 105 K), 52.58 is TlBa2Ca2Cu3O9 (8 rows, none measured), and nearby x2 values are O- or doping variants of these. Across about 200 Hg-cuprate measurements the ceiling stays at 136 K, apart from two Hg-Pb records at 143 K. The GP route is unavailable (crash).
**Decision:** switch to lib/fixed_points, which ran cleanly in B1-B6. The list is 13 unmeasured Tl-2223 duplicates, 8 Tl-1223 duplicates, and 9 Tl-2223 near-variants (x2 58.32-59.03).
**Hyperparameters:** not applicable (list from work/design8.py; the first 30 of work/design8.json).
**Block length:** 30. That leaves 14 for a final block, either the rest of the list (Tl-2234/1234 variants) or siblings of anything surprising.
**Review:** max per stoichiometry. If something reaches 130 K or more, sample its remaining duplicates. Otherwise finish with the remaining 14 entries of design8.json.

Proposal: `{"script": "lib/fixed_points.py", "label": "tl_cuprates", "args": {"seed": 0, "points": [995, 1000, 1438, 1480, 1531, 2045, 2075, 2747, 3088, 4100, 4101, 6135, 8110, 3899, 4097, 4098, 4099, 5323, 6092, 17007, 17247, 1294, 1469, 8035, 8036, 996, 997, 998, 2192, 2466]}, "n_points": 30, "state": "restart", "warm_start": false}`


Runner: Block finished: 30 paid measurements; 14 remain.

## Block 9 (after 236 of 250 measurements)

**Budget:** 236 used / 250 total; 14 remaining.
**Question:** Which records get the last 14 measurements? The aim is to spend the budget on the highest-prior unmeasured records left.
**Mode:** exploit (prior-ranked records).
**Gain:** Very likely none against the 185 K incumbent (H2S, idx 20573). By domain prior, composition-based records stop near 165 K. The budget must still be spent, and these records are the best remaining candidates.
**Evidence:** Block 8 gave 30 Tl-Ba-Ca-Cu-O records, range 80-131 K. The best were 131 K (17007, Tl-1223), 129 K (1531, Tl-2223), 128.5 K (6135) and 125 K (4100). Every Tl-2223 and Tl-1223 duplicate is now measured, so no group at 130 K or more has unmeasured siblings left. No GP is available: gp_bo crashes in this sandbox.
**Decision:** restart lib/fixed_points under the new label tl_final, using the last 14 entries of work/design8.json. These are Tl-2234/1234 and near-variant records, all confirmed unmeasured by work/final9.py.
**Hyperparameters:** not applicable.
**Block length:** 14, the whole remaining budget. This is the final block.
**Review:** none; the campaign ends. Best remains 185 K unless one of these records exceeds it.

Proposal: `{"script": "lib/fixed_points.py", "label": "tl_final", "args": {"seed": 0, "points": [2927, 2928, 2467, 7235, 2196, 4028, 7234, 2195, 2194, 986, 2193, 2042, 2043, 4096]}, "n_points": 14, "state": "restart", "warm_start": false}`


Runner: Block finished: 14 paid measurements; 0 remain.
