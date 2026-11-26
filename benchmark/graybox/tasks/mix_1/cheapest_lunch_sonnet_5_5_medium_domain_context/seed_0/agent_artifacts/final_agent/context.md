Pool problem, 350 dims, minimize viscosity. Run skill/aopt from workspace root (parent of .agentoptimizer). gp_bo/turbo crash - don't use.
Best 0.4344 (row 137). Blocks: random20; npgp 40; continue 60; continue 50 (best .434); block5 continue 40 planned.
work/npgp.py custom numpy GP EI on PCA-20, npc=20. work/a.py prints per-block stats.
Review: if stalled, tune npc/xi; final block uses remaining 40.
Block6 final: npgp continue 40. Block5 no improvement (best .4344).
