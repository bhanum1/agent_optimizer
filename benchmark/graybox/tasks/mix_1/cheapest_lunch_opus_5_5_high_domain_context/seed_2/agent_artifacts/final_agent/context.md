# Context (call 9 written)
Problem: pool of 5595 binary mixtures (candidates.csv rows = indices). Minimize y = dynamic viscosity (cP), at room temperature.
350 vars: x0..x215 comp1 RDKit descriptors, x217..x433 comp2, x434/x435 mole fractions. 273 components, 804 pairs, ~7 fractions/pair.
Deterministic, budget 250, parallel 1. Run `skill/aopt status|validate` from the workspace root. Scripts:
 `cd .agentoptimizer; ../skill/aopt python work/X.py`. Run check.py from the root.
botorch fitting HANGS in the sandbox -> use numpy models. A full litre fit/predict takes ~20 s.

Best: 0.434 at cand 1352 (butanone 0.83 + pentanol isomer 0.17). Next: 225 0.441, 4959 0.445, 4072 0.448. No improvement since B1.
The pool is effectively exhausted. The lowest unmeasured posterior mean is 0.541 (old pairs, sd 0.03), and new known pairs are predicted at 0.65 or above.
The unmeasured fractions of the best pairs all have more of the viscous component (work/c9.py).

MODELS: litre2/3/4 (lit-offset GP; ETA = literature pure viscosities). litre4 = known_only + new_pairs_only + tpi acquisition.
ML hypers at 215: tk 0.2, tu 0.8, pair 0, sn 0.03, po 0.1.
Scripts: work/b7chk.py and work/b8chk.py (prospective residuals for B7 and B8), work/rank7.py (posterior ranking -> rank7.pkl),
 work/tail7.py, work/c9.py (fractions of the best pairs).
B7 prospective: RMSE 0.147, residuals -0.17..+0.63 (heavy tail only on the positive side).

Hypotheses (all confirmed): the lit prior is reliable; unknown components are viscous; the pool floor is ~0.43-0.54.

Blocks:
- B1 screen 28: best 0.434. B2 mixre EI 30, B3 litpick 12, B4 litre EI 30, B5 litre2 PI 40, B6 litre3 PI 40: none.
- B7 litre4 tpi 35: none (best 0.595).
- B8 litre4 continue 20: none (best 0.644, cand 1813). Prospective RMSE 0.105, z-RMS 0.84, residuals -0.17..+0.30. No surprise.
- B9 litre4 continue 15 (final, whole remaining budget): planned.

Next call: the budget should be exhausted. If any points remain, continue litre4.
