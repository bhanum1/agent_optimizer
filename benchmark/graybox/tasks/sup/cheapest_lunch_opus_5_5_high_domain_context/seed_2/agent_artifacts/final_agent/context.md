# Context (call 11, final block)
Problem: pool of 21263 materials, 81 descriptors; y = -Tc (minimized); deterministic; budget 250.
BEST: 185.0 K at candidate 20573 (H2S). The domain prior says nothing else in the pool is likely to beat it.
B10 (tail10, 24 points): max 143 K (3985, 5200, Hg-Pb weighted mean mass 54); the kNN picks scored 110-118 K.
B11 (final11, 23 points, the last block): neighbours of 3985/5200 plus the nn8.py top 13 kNN picks.
Scripts: work/nn8.py (kNN ranking of unmeasured candidates), work/nb11.py (neighbours of chosen points), work/hyd4.py and hyd5.py (hydride decoding).
