# Lab fluid mixtures

The task is a pool of every heat-transfer fluid mixture that sdagent's lab measured
with μ, c_p and ρ: 1,506 rows from campaigns 1–11, over 34 fluids (12 pure, 216
binary, 1,278 ternary). No composition appears twice. Campaign 12 (975 rows) is
left out because it has no c_p. Campaigns 1–10 and campaign 11 are combined, even
though sdagent keeps their pure-component bases separate for excess calculations.
Each row here is a direct measurement.

- **Features:** the volume-fraction-weighted mean of the components' RDKit 2D
  descriptors. That is RDKit's full descriptor set minus those that are constant or
  non-finite across the 34 fluids, which leaves 138. The names are in
  `provenance.json`. The matrix has rank 33, because it is a linear image of the
  composition simplex.
- **Objective:** `y = log10(μ / (c_p·ρ)²)`, the laminar pumping-power figure of merit
  at fixed heat duty, in m³·s·K²/J. μ is in Pa·s (converted from cP), c_p in J/(kg·K)
  and ρ in kg/m³ (converted from g/mL). Lower is better.
- **Optimum:** `f_opt = −15.998`, a 25/75 binary from campaign 11. The 15 best rows
  all come from campaign 11.

**Difficulty.** The landscape is a smooth single funnel:
- fitness-distance correlation 0.85, nearest-neighbour correlation 0.92;
- 52 local minima in the 10-nearest-neighbour graph;
- the top rows are near-ties: the best and second best differ by 0.015, on a range
  of 1.24.

Random search finds the optimum in 3.3% of 50-evaluation runs. An ARD GP fit on 40
random rows ranks the rest with Spearman 0.89.

The measurements are proprietary (sdagent `docs/data.md`) and this repository is
public. So the inputs and the built pool live only in git-ignored
`external/fluid_mixtures/`:

| File | Source |
|---|---|
| `measurements.csv` | sdagent `data/processed/measurements.csv` (on the work machine) |
| `fluid_library.csv` | sdagent `data/processed/fluid_library.csv`, used for the SMILES by CID |
| `data.csv` | the built pool, written by `build.py` |

Only `build.py` and `provenance.json` are tracked. `provenance.json` records the
input and output hashes and the RDKit version. RDKit isn't in this repository's
environments, so rebuild the pool with sdagent's:

```bash
~/Documents/Work/Grad/sdagent/.venv/bin/python data/fluid_mixtures/build.py
```

The task is registered as `fluid_mixtures`, built as `CSVPoolTask(..., maximize=False)`
in `src/agentoptimizer/tasks/csv_pool.py`. Each evaluation picks one untested mixture by
anonymous index. The agent sees only the feature matrix; CIDs, compositions and
labels stay private.

## Benchmark at budget 50

The baseline is `sobol_gp_bo` in `scripts/run_campaign.py`. Its first 10% of the
budget (5 rows) comes from scrambled Sobol points in the min–max-scaled feature box,
each mapped to the nearest row not yet chosen. The rest comes from pool GP-BO with
BoTorch's default `SingleTaskGP`: an ARD RBF kernel with the dimension-scaled
length-scale prior, and LogEI over every unused row.

| Run | Evaluation at which it found the optimum | Evaluation at which it reached the top 1% |
|---|---|---|
| `sobol_gp_bo`, seeds 0–4 | 22, 24, 16, 16, 26 (median 22) | 13, 7, 14, 9, 25 (median 13) |
| agent (Claude Opus, high effort), seed 0 | 24 | 20 |

All six runs end on the optimum. The agent (run `run-3f330e4bfde0`, audit passed)
spent 12 evaluations on a fixed design, then ran its own PCA-GP optimizer for
5 blocks.

```bash
.venv/bin/python scripts/run_loop.py --task fluid_mixtures --budget 50 --seed 0
for s in 0 1 2 3 4; do
  .venv/bin/python scripts/run_campaign.py --task fluid_mixtures --optimizer sobol_gp_bo \
    --seed $s --budget 50 --outdir runs/fluid_bench_50
done
```
