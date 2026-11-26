# Context (call 9, final block)
Problem: pool of 21263 superconductors with 81 composition descriptors. y = -Tc (K), minimized. Budget 250, deterministic.
Best: 185 K at idx 20573 (H2S). Next best: 143 K Hg-Pb-1223.
- lib/gp_bo CRASHES in this sandbox (torch PermissionError). Use fixed_points.
## Blocks
B1 26 pts -> 133; B2 30 -> 135.7; B3 32 no gain; B4 29 -> 185 (H2S); B5 38, B6 51 no gain; B7 crash, 0 pts;
B8 30 Tl cuprates, max 131 K (17007), no gain; B9: final 14, the rest of design8.json (label tl_final). Budget is then exhausted.
