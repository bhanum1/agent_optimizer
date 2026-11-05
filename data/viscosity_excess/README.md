# Viscosity excess pool

The task is a pool of every heat-transfer fluid mixture that sdagent's lab measured
for viscosity: 2,458 binaries and ternaries from campaigns 1–12, over 40 fluids. The
goal is the mixture whose viscosity falls furthest below ideal mixing:

    y = ln μ_mix − Σ_i φ_i ln μ_i      (volume fractions φ, μ in cP; lower is better)

This follows sdagent's definition (`src/utils/excess.py`): the chord is drawn through
volume fractions, and viscosity is mixed on the ln scale. The pure values come from
two sources:
- **Campaigns 1–10:** the basis sdagent recovers exactly from `EV_lab_results.csv`
  (`benchmarking/pures.py`).
- **Campaigns 11–12:** their measured pure rows.

For the 8 fluids that both cover, the two pure ln μ values differ by at most 0.03,
against excesses of 0.4–0.65 at the top of the pool. Pure rows are not candidates.

- **Features:** the volume-fraction-weighted mean of the components' RDKit 2D
  descriptors (138, the same scheme as `fluid_mixtures`).
- **Optimum:** `f_opt = −0.646`, a ternary. The second best is −0.524. 96% of mixtures fall below ideal (median −0.13).

**Difficulty.** This target is harder than `fluid_mixtures`:
- fitness-distance correlation 0.33 (against 0.85), nearest-neighbour correlation 0.59;
- a GP fit on 40 random rows ranks the rest with Spearman 0.53, and its predicted
  best is typically the 73rd-best row;
- random search finds the optimum in 2% of 50-evaluation runs and the top 1% in 40%.

The inputs are proprietary, so they live in git-ignored `external/fluid_mixtures/`.
`export_excess.py` must run inside sdagent, where it needs the pure-basis recovery and
the measurement table, and it writes `excess_rows.csv`. `build.py` then runs with an
RDKit environment and writes `viscosity_excess.csv` and the tracked
`provenance.json`:

```bash
# on the machine with sdagent's data
cd ~/Documents/sdagent && .venv/bin/python <repo>/data/viscosity_excess/export_excess.py \
  > <repo>/external/fluid_mixtures/excess_rows.csv
# from this repository's root
~/Documents/Work/Grad/sdagent/.venv/bin/python data/viscosity_excess/build.py
```

## Benchmark at budget 50

The baseline is `sobol_gp_bo`: 5 Sobol rows (10% of the budget), then pool GP-BO with
BoTorch's default ARD GP and LogEI. It is described in
[`fluid_mixtures`](../fluid_mixtures/README.md#benchmark-at-budget-50).

| Run | Found the optimum at | Reached the top 1% at | Best after 10 | Best after 20 | Best after 30 |
|---|---|---|---|---|---|
| `sobol_gp_bo` seed 0 | 17 | 15 | −0.277 | −0.646 | −0.646 |
| seed 1 | 37 | 31 | −0.277 | −0.306 | −0.306 |
| seed 2 | 47 | 18 | −0.277 | −0.495 | −0.495 |
| seed 3 | 33 | 28 | −0.215 | −0.215 | −0.524 |
| seed 4 | 35 | 16 | −0.260 | −0.495 | −0.524 |
| **baseline median** | **35** | **18** | | | |
| agent (Claude Opus, high effort), seed 0 | 34 | 16 | −0.313 | −0.495 | −0.495 |

All six runs end on the optimum. The agent (run `run-eec282b2b03d`, audit passed) took
these steps:
1. a k-means design of 12 rows on 20 principal components;
2. its own GP on the top 10 components, with a rank transform of y, using LogEI for
   20 evaluations and then UCB for 8;
3. two blocks of 5 hand-picked neighbours of the best rows.

The Sobol initial rows overlap across seeds (rows 1753, 1654 and 1803 recur). In 138
dimensions, many Sobol points map to the same central "hub" rows.
