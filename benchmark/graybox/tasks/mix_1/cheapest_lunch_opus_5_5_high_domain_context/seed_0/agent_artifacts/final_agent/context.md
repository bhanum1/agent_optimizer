# Context (call 10 written)
Problem: pool of 5595 binary mixtures (indexed pool; pick a candidate row index). Minimize dynamic viscosity y (cP).
Deterministic, budget 250, no replicates. Variables = comp1 descriptors, comp2 descriptors, mole fraction (x434 = fraction of comp1).
273 molecules (my ids), about 804 pairs. No pair has two low-viscosity components. In each pair, the most favourable low-eta fraction is nearly always measured already.
Run aopt from the workspace root: `skill/aopt ...` (or ../skill/aopt from .agentoptimizer). Work scripts chdir themselves.
Do NOT print decisions.md (huge args); use tail -c or grep.
Files: work/mols_prior.pkl (per-mol prior ln eta, MolWt, known flag, fr_Al_OH); work/pairs_prior.pkl (i1, i2, mole_fraction_1, prior).
Guessed identities (prior only): 108 MTBE, 63 diethyl ether, 148 pentane, 20 MeCN, 87 acrylonitrile, 8 hexane, 11 heptane, 56 acetone,
31 butanone, 13 ethyl acetate, 36 cyclopentane, 47/40 esters, 24 methanol, 3 ethanol; alcohols 1,51,33,34 butanols, 21/22 propanols, 35 pentanol,
7 hexanol, 76 heptanol, 12 octanol, 94 decanol, 97 tert-amyl alcohol.

## Results (204 measured)
Best 0.4344, cand 1352 (butanone 0.83 + 97). Then 225 0.441, 4959 0.445, 4072 0.448, 4109 0.465. No gain in B4–B8 (B7 best 0.539, B8 best 0.578).
Below 0.5 = 60–85% of a ≤0.42 cP solvent plus an alcohol. Between-source scatter is about 0.1 ln.
The model now gives each candidate PI ≤0.7%; the pool looks nearly exhausted.

## Hypotheses
- Fitted mol offsets: known non-alcohol ≈ -0.28; known alcohols (MW>50) ≈ -0.63 ± 0.37; unknown (descriptor prior) sd ≈0.3–0.7.
- hbo7 calibration: trained on B1–6 → B7: rmse 0.24, z mean +0.35, z sd 1.21. Trained on B1–7 → B8: rmse 0.125, z mean +0.05, z sd 0.57, Spearman 0.83 (well calibrated and conservative).
- Model predicts all unmeasured candidates ≥0.598; EI is about 1e-4 everywhere; PI top picks are about 0.3% each.
- No index/source structure: LOO residual neighbour correlation 0.25 at gap≤5 (same pair), about 0 beyond that (work/idxres.py, 90 s; work/loo_res.npy).
- Unmeasured low-prior mols (218, 114, 15, 149, 96, 216) only pair with viscous alcohols; predicted ≥0.79.

## Models
- work/hbo7.py (CURRENT, label hbo7): hierarchical Bayesian linear model, exact posterior, EI on ln y, refit each step.
  Args in work/hbo7_args.json (made by work/mkargs7.py). About 1 s per step.
- work/t7.py '<json overrides>': warm-starts hbo7 on all data and fantasizes 25 picks (21 s).
- work/cal8.py '<json overrides>': forward calibration of B6 and B7 (train on earlier blocks). Change tb list for B8.
- work/b7.py: block table + top 15 overall + per-block min/median (edit block==7 → 8).
- work/toppairs.py: all fractions and measured status for the top-20 points' pairs.
- work/cov7.py: coverage of low-prior molecules. work/pairs_of.py <mol...>: pairs/fractions/y for a molecule.

## Blocks
- B1 prior_probe (fixed, 24): best 0.448.
- B2 structbo (EI, 20): no gain.
- B3 structbo2 (EI, 25): 0.448 → 0.434 (1352).
- B4 structbo3 (25), B5 structbo4 (30): no gain.
- B6 addpi6 (fixed top-PI, 25): no gain.
- B7 hbo7 (custom EI, warm start, 30): no gain; best 0.539. Unknown-molecule picks were mostly poor (279 1.41, 4466 1.21).
- B8 hbo7 continue (25): no gain; best 0.578 (537).
- B9 hbo7pi (restart from the same args with acq=pi, warm start, 24): no gain; best 0.566 (892), median 0.626, consistent with the model.
- B10 hbo7pi continue (22, FINAL): pending. Picks from the t7 simulation: 450, 4543, 529, 5206, 4853, 4686, 3972... PI about 0.15% each.

## Next review
- Budget should be fully spent after B10. If it is called again, report the best value; nothing remains to decide.
- work/ideal10.py ranks unmeasured candidates by the ideal-mixing prior (best unmeasured 0.494, unreliable).
