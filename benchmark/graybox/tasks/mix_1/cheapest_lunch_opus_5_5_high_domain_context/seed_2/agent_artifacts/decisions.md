# Decisions

## Block 1 (after 0 of 250 measurements)

**Budget:** 0 used / 250 total; 250 remaining.
**Question:** Which low-viscosity components (small, non-H-bonding solvents) actually give the lowest mixture viscosity, and does a chemistry prior rank candidates correctly?
**Mode:** both (prior-guided screening, mostly in the region expected to be best).
**Gain:** Probably a strong incumbent right away (expect well below 1 cP if the prior is right). It also shows which components to exploit and whether to rely on a prior-informed or a learned surrogate.
**Evidence:** There are no measurements yet. The pool has 5595 binary mixtures: 273 distinct components, 804 distinct pairs, and about 7 mole fractions per pair (x434 = x1, x435 = x2). The space has 350 descriptor dimensions, which is too many for a GP fitted on a few points. Domain prior (an assumption, not data): low viscosity comes from small, non-H-bond-donor, low-polarity molecules (alkanes, ethers, ketones, esters, nitriles). Each extra donor, ring or unit of polarity raises viscosity. Water and H-bond networks raise it too. As a first guess, mixtures mix roughly log-linearly in mole fraction. I wrote a crude heuristic, ln(eta) ~ -2.3 + 0.011*MW + 1.3*HDonors + 0.25*Rings + 0.012*TPSA (+1.9 for water), mixed it log-linearly, and ranked the pool with it. The heuristic is known to underrate very polar aprotic components such as DMSO and DMF.
**Decision:** Experiment with fixed_points. For each of the 24 components ranked lowest by the heuristic, I take the candidate where that component makes up at least 50% of the mixture and the predicted mixture value is lowest. Each pick is a different pair. I add 4 calibration points at the heuristic's 30/55/80/95% quantiles, on new pairs. Outcomes: if the low-prior points are clearly lower than the calibration points, the prior holds, and I exploit the best components and pairs with a surrogate on low-dimensional, mole-fraction-weighted features (custom pool BO). If they are not, the prior is wrong, and I switch to a data-driven surrogate with broader exploration.
**Hyperparameters:** fixed_points takes only the list of points: indices [242, 3233, 1414, 2915, 1770, 2851, 1037, 1352, 2835, 4796, 660, 566, 2085, 1837, 2540, 4072, 2305, 153, 3357, 5082, 849, 3171, 4269, 1615, 4210, 1826, 5393, 5515] (work/select2.py, work/final1.py).
**Block length:** 28 measurements (11% of the budget). That is enough to cover about 20 distinct low-viscosity components plus calibration before fitting a model.
**Review:** Rank the components by measured y. Check the heuristic's correlation with y. Fit a GP or ridge model on a few mixture-weighted descriptors and check leave-one-out accuracy. Then design a pool BO focused on pairs of the best components.

Proposal: `{"script": "lib/fixed_points.py", "label": "screen", "args": {"points": [242, 3233, 1414, 2915, 1770, 2851, 1037, 1352, 2835, 4796, 660, 566, 2085, 1837, 2540, 4072, 2305, 153, 3357, 5082, 849, 3171, 4269, 1615, 4210, 1826, 5393, 5515], "seed": 0}, "n_points": 28, "state": "restart", "warm_start": false}`


Runner: Block finished: 28 paid measurements; 222 remain.

## Block 2 (after 28 of 250 measurements)

**Budget:** 28 used / 250 total; 222 remaining.
**Question:** Can a structured mixture model (component effects plus mixing terms) pick unmeasured candidates that beat 0.434 cP? Which components and pairs are worth exploiting?
**Mode:** both (model-guided exploitation with EI, which also explores components it is uncertain about).
**Gain:** Improve on the incumbent 0.434 cP (cand 1352: 0.83 butanone-like ketone (MW 72.1) with 0.17 C5 alcohol). Expect 0.35–0.43 cP if high-fraction mixtures of low-viscosity components exist with a good partner. The block also gives 30 more labels to recheck the model's LOO accuracy.
**Evidence:** Block 1 measured 24 low-prior picks at 0.43–1.07 cP and 4 calibration points at 1.0–3.96 cP. The prior ranks only weakly (Spearman 0.34). Pool structure matters: almost every pair contains a "hub" component (an alcohol such as propanol, butanol, pentanol or heptanol, or water). No pair has two low-viscosity, non-donor components. The lowest-viscosity solvents appear only at limited own fractions: the pentane-like component at ≤0.52 and the ether-like component at ≤0.54. So the minimum likely sits in high-fraction mixtures of a low component with a short alcohol. Observed values fit log-linear mixing with negative excess (e.g. ACN 0.87 + heptanol gives 0.487, the log-linear estimate is 0.49). Offline tests: GPs on engineered mixture features have poor LOO (RMSE 0.52–0.63 vs y-std 0.61 in log). The custom Bayesian random-effects mixture model has LOO RMSE 0.42, Spearman 0.46 and calibrated |z| of about 0.6. It is the best surrogate available. Its top predicted unmeasured candidates are at 0.48–0.51 cP with sd 0.2–0.3 in log units. EI first picks high-fraction octane-isomer mixtures and propanol/butanone pairs.
**Decision:** Switch from the fixed screen to a custom optimizer, work/mixre.py, warm-started with all 28 points. The model is ln y = x1·g(c1) + x2·g(c2) + x1x2·(e0 + e1|ΔHD| + e2·water + e3·asym) + x1x2·pair effect, where g(c) = w·descriptors + component random effect. The prior on w is centred on the chemistry heuristic, which is domain knowledge used as a prior. The posterior is exact (a GP with a structured linear kernel). Noise and component sd are chosen by marginal likelihood on a grid at every step. Acquisition is EI in log space over unused candidates. check.py passed.
**Hyperparameters:** tau_grid [0.3, 0.5, 0.8] (component sd, log units). sn_grid [0.05, 0.1, 0.2]. pair_sd 1.0 (excess per pair, scaled by x1x2 ≤ 0.25). xi 0. Descriptor prior sds 0.2–1.0 (in the file). On the current data, ML picks tau 0.5 and sn 0.05.
**Block length:** 30 measurements, about 14% of the remaining budget. That is enough for EI to probe several new components and refine the best pairs before I recheck model calibration.
**Review:** Did the best value improve, and which components and pairs gave the lowest values? Recompute the LOO of mixre on 58 points. If EI wastes points on high-uncertainty, high-mean components, switch to TS or a lower-confidence-bound rule, or restrict to high-fraction candidates of the best components. Consider adding a residual GP over the mole fraction within the best pairs.

Proposal: `{"script": "work/mixre.py", "label": "mixre", "args": {"seed": 0, "acquisition": "ei", "xi": 0.0, "tau_grid": [0.3, 0.5, 0.8], "sn_grid": [0.05, 0.1, 0.2], "pair_sd": 1.0}, "n_points": 30, "state": "restart", "warm_start": true}`


Runner: Block finished: 30 paid measurements; 192 remain.

## Block 3 (after 58 of 250 measurements)

**Budget:** 58 used / 250 total; 192 remaining.
**Question:** Does a literature-informed log-linear mixing model (known pure viscosities plus a fitted excess term) rank candidates better than the descriptor-based mixre model? Do any untested high-fraction pairs of the lowest-viscosity components beat 0.434?
**Mode:** both. The block exploits the top-ranked candidates and also validates the new surrogate prospectively.
**Gain:** About a 30% chance per top candidate of beating 0.434, so a good chance that at least one of the 12 does. Candidates are predicted at 0.47–0.57, with an LOO sd of about 0.14 in log units. The results also tell me whether to build the next optimizer on the lit model.
**Evidence:** Block 2 (mixre EI, 30 points) did not improve the best value; its points fell between 0.465 and 1.16. mixre now has LOO RMSE 0.33 (log), rho 0.60, and it is mean-reverting: it predicts the best points at 0.49–0.55. New offline analysis uses domain knowledge as a prior, not as data. I identified components from their descriptors and gave them approximate 25 °C literature pure viscosities (work/lit.py), e.g. ACN 0.34, butanone 0.40, MTBE 0.34, pentanol 3.6. The model is ln y = Σ x ln eta + x1x2·(b0 + b1(1−2x_donor)) for alcohol/non-donor pairs, plus b2 for same-type pairs. On the 58 points it gets RMSE 0.129 and LOO RMSE 0.138, rho 0.89, with b0 = −1.27. Outliers: DMSO at 0.98 measured 3.40 against about 2.0 expected, which suggests the hidden measurement conditions vary. Pair-specific effects are large: butanone/pentanol-isomer pairs gave 0.434 at x = 0.83 but 0.554 at x = 0.74. The model implies the optimum is the highest fraction of the lowest-viscosity component, since adding alcohol never helps at the fitted b. Only 37% of the pool has both components identified. Unidentified non-donor components (C5/C6 esters, aromatics, chlorinated solvents) are about 0.5 cP or above, so they are unlikely to beat 0.43. The pool floor is probably about 0.40–0.45.
**Decision:** Switch to a fixed experiment from the lit-model ranking (work/litrank.py, work/pick3.py), restricted to new pairs, at most 2 per low component. I skipped near-duplicates of already-measured pairs, whose outcome is predictable. Points: 3078 (vinyl acetate/methyl acrylate 0.948 + decanol), 237 and 225 (ACN 0.76 + pentanol, ACN 0.73 + butanol), 4959 (heptane 0.82 + pentanol), 685 and 691 (cyclopentane 0.79 + butanol, 0.81 + pentanol), 3061 (vinyl ester 0.86 + heptanol), 4972 and 159 (ethyl acetate 0.80 + pentanol, 0.87 + octanol), 1345 (butanone 0.74 + another pentanol isomer), 4261 (THF 0.80 + hexanol), 4605 (C6 ether 0.73 + pentanol). If measured values track the predictions (RMSE ≲0.15), I build a lit-offset model with pair and component random effects for the rest of the campaign. If they do not, I fall back to mixre and component-level exploitation.
**Hyperparameters:** fixed_points takes only the index list. The lit model's coefficients were fitted by least squares on the 58 points, excluding the DMSO outlier.
**Block length:** 12 measurements. That is enough to check the lit model prospectively and to cover the most promising untested pairs, while keeping the budget for a model-guided phase.
**Review:** Compare measured values with litpred. Did any beat 0.434? Fold residuals into a hybrid optimizer (lit offset + mixre random effects) and test its LOO.

Proposal: `{"script": "lib/fixed_points.py", "label": "litpick", "args": {"points": [3078, 237, 225, 4959, 685, 691, 3061, 4972, 159, 1345, 4261, 4605], "seed": 0}, "n_points": 12, "state": "restart", "warm_start": false}`


Runner: Block finished: 12 paid measurements; 180 remain.

## Block 4 (after 70 of 250 measurements)

**Budget:** 70 used / 250 total; 180 remaining.
**Question:** The best identified pairs are now exhausted at their highest low-viscosity fractions. Can a model-guided search with the literature prior, extended to most components, find pool entries below 0.434 cP? Should those come from component uncertainty (unknown or partly known components) or from pair and source variation?
**Mode:** both. EI exploits the low predicted means and explores the uncertain components.
**Gain:** Under the model, P(at least one of the top-25 EI candidates beats 0.434) is about 0.8. That is probably optimistic: realistically I expect 0.40–0.43 if anything improves. The block also gives 30 points to recheck calibration of the extended prior.
**Evidence:** Block 3 tested the lit model prospectively on 12 points. Log residuals were +0.10, +0.10, -0.09, -0.09, +0.09, -0.01, +0.07, -0.02, +0.16, +0.11, +0.10, +0.13, so RMSE about 0.10 with a slight positive bias. The prior holds, but no point beat 0.434; the closest were 225 (ACN + butanol) at 0.441 and 4959 (heptane + pentanol) at 0.445. Every top measured pair is already at its highest available low-viscosity fraction, and the lit model's best unmeasured identified candidates are predicted at 0.47–0.51. I extended the literature pure-viscosity table (domain knowledge, about 25 °C; the context says room temperature): amines (diethylamine 0.32, propylamines 0.36), C5/C6 esters, pentanone, TCE, xylenes, chlorinated solvents, glycols, glycol ethers, aniline and acids. It now covers 184 of 273 components and 77% of the pool, up from 37%. The other components get a ridge QSPR fitted on the identified ones (RMSE 0.52 log). The new custom model work/litre.py is ln y = Σ x_i (ln eta_i + u_i) + xx·(b·[mix, mix(1−2x_donor), same]) + const + pair effect. It uses exact Gaussian posteriors with hyperparameters chosen by marginal likelihood (currently tau_known 0.2, tau_unknown 0.4, pair 0.3, sn 0.06). On 70 points: LOO RMSE 0.156 (0.130 without the DMSO outlier), Spearman 0.85, median |z| 0.54. The lowest posterior means among unmeasured candidates are 0.48–0.55 with sd about 0.05. The top EI candidates are C6 ether (di-n-propyl-ether-like, 0.9 fraction) + glycol ether (pred 0.49, sd 0.19), more ether/glycol-ether pairs, and isopropylamine and diethylamine pairs.
**Decision:** Switch to the custom optimizer work/litre.py (literature offset + component, pair and coefficient random effects, EI in log space), warm-started with all 70 measurements. check.py passed with --warm 70.
**Hyperparameters:** xi 0, because the incumbent margin matters and EI already explores through the posterior sd. Hyperparameter grids: tau_k [0.1, 0.2, 0.35], tau_u [0.4, 0.8], pair [0.3, 0.7, 1.2], sn [0.03, 0.06, 0.12]; all are re-selected by marginal likelihood at each step. Coefficient prior: b ~ N([-1, 0, 0], 0.7²), offset N(0, 0.2²).
**Block length:** 30 measurements (17% of the remainder). That is enough for EI to probe about 10 new components and pairs and to update their effects, while leaving 150 for whatever the review suggests.
**Review:** Did the best value improve? What are the prospective residuals of the extended-prior components (amines, esters, glycol ethers)? Is EI wasting points on high-sd unknowns? If so, lower tau_u or switch to 'mean'/LCB. Also check whether pair effects look larger than the selected 0.3.

Proposal: `{"script": "work/litre.py", "label": "litre", "args": {"seed": 0, "acquisition": "ei", "xi": 0.0, "tau_k_grid": [0.1, 0.2, 0.35], "tau_u_grid": [0.4, 0.8], "pair_grid": [0.3, 0.7, 1.2], "sn_grid": [0.03, 0.06, 0.12]}, "n_points": 30, "state": "restart", "warm_start": true}`


Runner: Block finished: 30 paid measurements; 150 remain.

## Block 5 (after 100 of 250 measurements)

**Budget:** 100 used / 250 total; 150 remaining.
**Question:** The model now says the pool is nearly exhausted below 0.434. Where is a lucky improvement most likely, and how do I stop EI from spending points on hopeless high-uncertainty components?
**Mode:** exploit. This is a probability-of-improvement search over new pairs, and it also tests whether pairs carry a source-level offset.
**Gain:** Small, stated honestly. Under the refitted model the summed probability of improvement over all unmeasured candidates is about 0.4, and about 0.33 for the top 40 picks. So I expect perhaps a 25–35% chance of beating 0.434, likely by only a few percent. Every alternative I checked does worse.
**Evidence:** Block 4 (litre EI, 30 points) did not improve the best; its values ran 0.482–1.91.
- **Prospective accuracy:** log RMSE 0.24 and z-RMS 0.99, so the model is well calibrated. Low-sd predictions for unmeasured fractions of already-measured pairs were nearly exact (4108 predicted 0.481, measured 0.487; 261 predicted 0.496, measured 0.482; 224 predicted 0.485, measured 0.504).
- **What EI picks now:** the top EI points are high-mean, high-sd components with no literature value (predicted 1.2–1.4 cP, sd 0.33). They have essentially no chance of beating 0.434. Under the model, P(any of the top-25 EI points beats 0.434) is only 0.02.
- **The unidentified components:** I reviewed all 89. They are ionic-liquid-like species, long alkanes and esters, halobenzenes, bromoform, tetrachloroethane, tetramethylurea and similar. None is plausibly below 0.5 cP, so domain knowledge argues against exploring them.
- **Simulation:** I drew pool truths from the posterior and ran EI, PI, mean, LCB and an inflated-variance PI for 150 steps each. In both draws the true minimum among unmeasured candidates was above 0.434 (0.475 and 0.462), and no method improved. I stopped after 2 draws (about 160 s each). They confirm the floor but cannot rank the methods.
- **New model term:** marginal likelihood favours a per-pair constant offset with sd 0.1 (a source/temperature shift per data set). The best ML with it is 136.5 (with the xx-scaled pair term set to 0, tau_k 0.1); the best without it (pair offset sd 0) is 135.1. This is what makes an untested pair able to come in lower than predicted. Measured fractions of known pairs are predicted tightly, so they rarely improve.
- **PI picks:** PI under this model favours new pairs with low means: 5016 and 5015 (diglyme-like + C6 ether at 0.8), 5030 (glycol ether + C6 ether), 4968 (ethyl-acetate-like + C6 ether, 0.82), 5500 (ACN 0.8 + a C17 amide/amine), 402 (C8 alkane 0.8 + a C4 unsaturated alcohol), 4978 (propanol + ethyl acetate) and 4119.
- **Domain knowledge:** the literature pure viscosities (the ETA table) remain the prior; data have confirmed them.
**Decision:** Adjust the method. I built work/litre2.py, which is litre plus the per-pair constant offset (po) and a 'pi' acquisition: P(log y < log best − xi), computed with the posterior sd plus noise. Restart, warm-started with all 100 measurements; check.py passed with --warm 100. Hypothesis tested along the way: if new pairs scatter about ±0.1 around their predictions (po ≈ 0.1 holds), PI's focus on new low-mean pairs is right. If they land on their predictions (po → 0), the pool floor is confirmed, and the rest of the budget goes to the lowest-mean candidates.
**Hyperparameters:** xi 0, since any improvement counts. Grids, all re-selected by marginal likelihood at every step: tau_k [0.1, 0.2, 0.35], tau_u [0.4, 0.8], pair (xx-scaled) [0, 0.3, 0.7] (0 is added because ML now prefers it), sn [0.03, 0.06, 0.12], po [0, 0.05, 0.1, 0.15]. The coefficient prior is unchanged from litre.
**Block length:** 40 measurements. PI's mass is concentrated in its top ~40 (0.33 of 0.40), and 40 is enough to check whether po holds before choosing how to spend the last 110.
**Review:** Did anything beat 0.434? Check prospective residuals on new pairs: is their spread about 0.1, and is the fitted po still 0.1? How does the summed PI of the remaining pool look? If it is exhausted, consider the high-fraction, low-eta sweep of unmeasured fractions of the best pairs, or (only if the data suggest it) relax the monotonic-in-fraction form.

Proposal: `{"script": "work/litre2.py", "label": "litre2", "args": {"seed": 0, "acquisition": "pi", "xi": 0.0, "tau_k_grid": [0.1, 0.2, 0.35], "tau_u_grid": [0.4, 0.8], "pair_grid": [0.0, 0.3, 0.7], "sn_grid": [0.03, 0.06, 0.12], "po_grid": [0.0, 0.05, 0.1, 0.15]}, "n_points": 40, "state": "restart", "warm_start": true}`


Runner: Block finished: 40 paid measurements; 110 remain.

## Block 6 (after 140 of 250 measurements)

**Budget:** 140 used / 250 total; 110 remaining.
**Question:** The pool looks exhausted below 0.434. How do I spend the rest so that each point still has a real, if small, chance of improving the best?
**Mode:** exploit. This is probability of improvement restricted to candidates whose two components both have literature viscosities.
**Gain:** Small, stated honestly. The best known-only candidates have P(improve) of about 1–1.6% each, and the top 40 sum to about 0.25 under the model. The model's new-pair sd (0.16) is wider than the residuals actually seen (about 0.1), so the real chance is probably lower. Even so, these are the only points with non-negligible probability.
**Evidence:** Block 5 (litre2 PI, 40 points) did not improve the best value. Of its 40 picks, 24 had a component with no literature value; they measured 0.59–6.15, mostly above 1, and their prospective log residuals reached +1.1 to +1.4 (ACN with C17 amides, 2638). The 16 picks with both components known measured 0.51–0.77, with residuals from −0.10 to +0.27. Overall prospective RMSE was 0.39 (z-RMS 2.2), but it is about 0.1 for known-component new pairs. The ML hyperparameters at 140 points are tau_k 0.1, tau_u 0.8, pair 0, sn 0.03, po 0.15, so the per-pair source offset is supported. I listed every low-viscosity component (eta ≤ 0.42: pentane, diethyl ether, hexane, acetone, diethylamine, ACN, acrylonitrile, MTBE, methyl acetate, heptane, butanone, …) with all its pairs. Every pair whose highest available low-component fraction could reach about 0.45 has already been measured there. The lowest posterior mean among unmeasured candidates is 0.539 (sd 0.03, old pairs). The lowest among new pairs is about 0.62 (sd 0.16). Only 21 groups of duplicate feature rows exist, and none sits near the optimum. Domain knowledge used: the literature pure-viscosity table, which the data confirmed in blocks 3–5, and my earlier review that unidentified components are viscous species (ILs, long-chain, halogenated) that cannot reach below 0.5 cP. Block 5 confirmed that review.
**Decision:** Adjust the method. work/litre3.py is litre2 unchanged, plus `known_only`: proposals are restricted to unmeasured candidates whose two components both have literature values (4188 free). It falls back to the full pool only if that set empties. Restart with a warm start from all 140 measurements. check.py passed with --warm 140. Expected first picks: 1404, 892, 3074, 3768, 537. Hypothesis tested along the way: whether new known pairs ever land 2+ sd below prediction. If one does, the per-pair offset is heavier-tailed than assumed, and the next block should concentrate on new pairs of the same component types.
**Hyperparameters:** These are the same grids as block 5, re-selected by marginal likelihood at every step, with xi 0 because any improvement counts. known_only is true, based on the block 5 evidence that unknown-component picks are wasted.
**Block length:** 40 measurements. That covers the top PI mass of the known new pairs, about 0.22 of 0.35, and leaves 70 for a final review.
**Review:** Did anything beat 0.434? Check the residual distribution of the new known pairs (tails beyond −0.2?). Is the remaining summed PI worth continuing with (state=continue), or should the end go to a 'mean' ranking with one candidate per new pair?

Proposal: `{"script": "work/litre3.py", "label": "litre3", "args": {"seed": 0, "acquisition": "pi", "xi": 0.0, "tau_k_grid": [0.1, 0.2, 0.35], "tau_u_grid": [0.4, 0.8], "pair_grid": [0.0, 0.3, 0.7], "sn_grid": [0.03, 0.06, 0.12], "po_grid": [0.0, 0.05, 0.1, 0.15], "known_only": true}, "n_points": 40, "state": "restart", "warm_start": true}`


Runner: Block finished: 40 paid measurements; 70 remain.

## Block 7 (after 180 of 250 measurements)

**Budget:** 180 used / 250 total; 70 remaining.
**Question:** The model says the pool is exhausted below 0.434. Which remaining candidates have the largest real tail chance of beating it, given the calibration we actually observed?
**Mode:** exploit (a tail search), with a running calibration check on new pairs.
**Gain:** Small, stated honestly. Under a heavy-tailed (t3) residual model with the sd corrected to the observed calibration, each pick has about 0.5–0.7% chance of improving, about 0.2 summed over 35 picks. That figure is probably optimistic. The realistic outcome is that the best stays at 0.434, but no better use of these points exists.
**Evidence:** Block 6 (litre3 PI, known components only, 40 points) did not improve the best value. Its best point was 1404 at 0.506 (amine 0.74 + C5 alcohol/ether).
- **Calibration:** prospective log RMSE was 0.125 with z-RMS 0.74, so the model's sd is about 25% too wide for new known pairs. Residuals ran from −0.22 to +0.34, and the positive tail is the heavier one.
- **Hyperparameters:** marginal likelihood at 180 points now chooses tau_k 0.2, tau_u 0.8, pair 0.3, sn 0.03, po 0.05. The per-pair source offset has shrunk from 0.15.
- **What is left:** summed Gaussian PI is 0.0017 over known-only candidates and 0.26 over candidates with unknown components. That 0.26 comes entirely from components with no literature value, predicted at 1.3–2.8 cP with sd about 0.6. Block 5 showed these are viscous (measured 0.6–6.2), and my earlier domain review found them physically implausible below 0.5.
- **Why litre3 PI misleads now:** with the inflated sd, its top picks are water-rich mixtures (0.97–0.99 water, predicted 0.96–1.0) and 0.94-fraction mixtures of a component predicted near 0.9. Mixtures like these physically cannot reach 0.43.
- **Lowest predictions:** the lowest new known pairs are predicted at 0.65–0.70 (corrected sd about 0.08). The lowest old pairs are at 0.54 with sd 0.03, which needs a 7-sd miss to improve.
- **Domain knowledge used:** the literature pure-viscosity prior, confirmed in blocks 3–6, and the view that the unknown components are viscous species.
**Decision:** Adjust the method. work/litre4.py is litre3's model unchanged with a new acquisition, 'tpi': the Student-t (df 3) probability of improvement with model sd × 0.75 (the observed calibration). It is restricted to never-measured pairs whose two components are both known, and candidates with posterior mean above 0.8 cP are ranked last. With one point per new pair, every pick tests a different source/pair offset. Restart, warm-started with all 180 measurements. check.py passed with --warm 180. Offline first picks: 3227, 3414, 4589, 2152, 5286, 4923 (predicted 0.65–0.78).
**Hyperparameters:** The model grids are the same as blocks 5–6 and are re-selected by marginal likelihood at each step. sd_scale 0.75 comes from block 6's z-RMS of 0.74. t_df 3 allows heavier-than-Gaussian tails for possible pair or source surprises. mean_max 0.8 excludes physically hopeless picks. xi is 0.
**Block length:** 35 measurements, half of the remainder. That is enough to see whether any new pair lands far below its prediction (residual below −0.3). If one does, the last 35 can concentrate on that pair's components. If not, they continue this ranking.
**Review:** Did anything beat 0.434? Is there a residual below −0.25, and which component class does it share? Recheck the calibration (z-RMS with the scaled sd). For the final 35, continue litre4, or concentrate on the class of any surprising pair.

Proposal: `{"script": "work/litre4.py", "label": "litre4", "args": {"seed": 0, "acquisition": "tpi", "xi": 0.0, "tau_k_grid": [0.1, 0.2, 0.35], "tau_u_grid": [0.4, 0.8], "pair_grid": [0.0, 0.3, 0.7], "sn_grid": [0.03, 0.06, 0.12], "po_grid": [0.0, 0.05, 0.1, 0.15], "known_only": true, "new_pairs_only": true, "sd_scale": 0.75, "t_df": 3.0, "mean_max": 0.8}, "n_points": 35, "state": "restart", "warm_start": true}`


Runner: Block finished: 35 paid measurements; 35 remain.

## Block 8 (after 215 of 250 measurements)

**Budget:** 215 used / 250 total; 35 remaining.
**Question:** Did the tail search on new known pairs turn up a pair far below its prediction, one that would justify concentrating the last points on its component class?
**Mode:** exploit (a tail search), with a final check for surprises.
**Gain:** Small, stated honestly. The chance of beating 0.434 is a few percent at most. No remaining candidate has a meaningful probability under any model I have, so this block keeps spending points where the small chance that does exist is largest.
**Evidence:** Block 7 (litre4 tpi, 35 new known pairs) did not improve the best value. Its best point was 4286 at 0.595.
- **Prospective accuracy:** log RMSE 0.147. Residuals ran from −0.17 (4838) to +0.63 (3414), and the tail is heavy only on the positive side. No residual fell below −0.25, so there is no surprise class to concentrate on.
- **Hyperparameters:** marginal likelihood at 215 points chooses tau_k 0.2, tau_u 0.8, pair 0, sn 0.03, po 0.1.
- **What is left:** the lowest posterior means among unmeasured candidates are 0.541–0.56, all on already-measured pairs with sd about 0.03 (1310, 848, 4874, 5081). Gaussian PI is effectively 0 for every known-component candidate.
- **Best pairs:** the unmeasured fractions of the top pairs (1352, 225, 4959, 4072) all have a higher fraction of the viscous component. They are predicted at 0.59–4.0 and cannot improve.
- **Domain knowledge used:** the literature pure-viscosity prior (confirmed again here; median residual is about 0) and the view that unknown components are viscous (confirmed in block 5).
**Decision:** Continue litre4 with state=continue. Its model refits by marginal likelihood at each step, and the acquisition is the Student-t tail probability over never-measured known pairs, one test per pair offset.
**Hyperparameters:** Unchanged; they are ignored under continue. sd_scale 0.75 and t_df 3 still match the observed residuals (narrow core, heavy positive tail).
**Block length:** 20 measurements. That leaves 15 for a last review: if a pair lands at least 0.3 log below its prediction, the remaining points can go to sibling fractions or related pairs. Otherwise litre4 continues to the end.
**Review:** Did anything beat 0.434, or come in with a residual below −0.25? If not, continue litre4 for the final 15.

Proposal: `{"script": "work/litre4.py", "label": "litre4", "args": {"seed": 0, "acquisition": "tpi", "xi": 0.0, "tau_k_grid": [0.1, 0.2, 0.35], "tau_u_grid": [0.4, 0.8], "pair_grid": [0.0, 0.3, 0.7], "sn_grid": [0.03, 0.06, 0.12], "po_grid": [0.0, 0.05, 0.1, 0.15], "known_only": true, "new_pairs_only": true, "sd_scale": 0.75, "t_df": 3.0, "mean_max": 0.8}, "n_points": 20, "state": "continue", "warm_start": false}`


Runner: Block finished: 20 paid measurements; 15 remain.

## Block 9 (after 235 of 250 measurements)

**Budget:** 235 used / 250 total; 15 remaining.
**Question:** Did block 8 turn up a new pair far below its prediction, one that would justify putting the last points on its siblings? It did not, so the remaining question is only where the tail chance of beating 0.434 is largest.
**Mode:** exploit (the final tail search).
**Gain:** Small, stated honestly. The chance of beating 0.434 is a few percent at most. These are the only points with any chance of improving, and an unspent point earns nothing.
**Evidence:** Block 8 (litre4 tpi, 20 new known pairs) measured 0.644–1.13. Its best was 1813 at 0.644, so the best value did not improve.
- **Prospective accuracy:** from the 180-point model, log RMSE was 0.105 with z-RMS 0.84. Residuals ran from −0.17 to +0.30, and none fell below −0.25. There is no surprise pair or component class to concentrate on.
- **What is left:** the model stays well calibrated, and the remaining low predictions are about 0.54 on old pairs (sd 0.03) and 0.65 or above on new pairs (sd about 0.1). Reaching 0.434 would take a miss of more than 4 sd, which only the heavy-tailed tpi ranking weights at all.
- **Domain knowledge used:** the literature pure-viscosity prior (median residual about 0 again) and the view that unknown components are viscous.
**Decision:** Continue litre4 (state=continue), as planned at the last review. It is the Student-t tail probability of improvement over never-measured pairs whose two components are both known, one point per pair offset.
**Hyperparameters:** Unchanged; they are ignored under continue. sd_scale 0.75 and t_df 3 still fit the observed z-RMS of 0.84 and the positive-skewed tail.
**Block length:** 15 measurements, the whole remaining budget. No answer from a shorter block could change the method in time to matter.
**Review:** The campaign ends after this block. The final best is the lowest feasible measurement, currently 0.434 at candidate 1352.

Proposal: `{"script": "work/litre4.py", "label": "litre4", "args": {"seed": 0, "acquisition": "tpi", "xi": 0.0, "tau_k_grid": [0.1, 0.2, 0.35], "tau_u_grid": [0.4, 0.8], "pair_grid": [0.0, 0.3, 0.7], "sn_grid": [0.03, 0.06, 0.12], "po_grid": [0.0, 0.05, 0.1, 0.15], "known_only": true, "new_pairs_only": true, "sd_scale": 0.75, "t_df": 3.0, "mean_max": 0.8}, "n_points": 15, "state": "continue", "warm_start": false}`


Runner: Block finished: 15 paid measurements; 0 remain.
