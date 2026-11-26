# Context (call 7, final)
- Pool MOF CO2 uptake; minimize y=-uptake. Best after B6: Mg idx 3656 9.32 (then Ca 4630 9.16).
- Shell cwd = workspace root; run `skill/aopt python .agentoptimizer/work/X.py`; scripts r7*.py use .agentoptimizer/ paths.
- Sweet spot: AE (Mg/Ca/Sr), POAVF .45-.6, dens .75-1.0, Di 4.5-6.5. kNN LOO Spearman 0.54.
- Blocks: B1 init20 6.41; B2 gpei 20 none; B3 15 ->8.04; B4 15 AE confirmed; B5 AE-GP 15 ->9.16; B6 knn6 10 ->9.32.
- B7 final7: fixed [7318, 7763, 2970, 5058, 5767], 5 points = remaining budget.
- Scripts: work/r6b.py (kNN/GP ranking of region), r7.py (descriptor neighbours), r7b.py (AE table), r7c.py (unmeasured AE list).
