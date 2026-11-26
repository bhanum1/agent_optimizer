# Context (after call 8, final block)
Problem: pool of 5595 binary mixtures (anonymous candidate indices), 350 descriptor columns. comp1 columns x0..x215;
comp2 column = comp1 column + 217. Mole fractions x434/x435. Minimize viscosity y (cP). Deterministic, budget 250.
Model: log10 y ~ latent pure viscosities (literature prior) + Grunberg-Nissan excess + pair effect (latent_ei3/latent_pi4).
Best: 0.4344 cP, idx 1352 (MEK 0.83 + C5-alcohol 0.17), block 2. Not improved since.

Status: the model is calibrated (block 7 z mean -.10, sd .84). There are no low-side clusters. The heuristic components' errors skew upward.
Labels: latent_pi4 (work/latent_pi4.py, downside-aware PI, su_acq .1, se_acq .03): blocks 6, 7, 8 (final 18, continue).
Scripts: work/rev8.py = block-7 z (hist[43:]). For block 8 use hist[68:] and block==8. work/loo6.py, low6.py, nb6.py.

Blocks:
1. explore 24: best 0.51. 2. latent_ei 30: 0.434. 3-7: no improvement (latent_ei2 30+40, latent_pi3 40,
   latent_pi4 43+25). 8. latent_pi4 continue, 18 points (final).
