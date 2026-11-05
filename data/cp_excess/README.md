# Heat-capacity excess pool

The task is a pool of the mixtures from sdagent's lab campaign 11, the cohort that
searched near-exhaustively. It covers 12 fluids: 157 of the 198 grid binaries and
1,131 of the 1,540 grid ternaries. The goal is the mixture whose heat capacity rises
furthest above ideal mixing:

    excess c_p = c_p,mix − Σ_i φ_i c_p,i      (volume fractions φ, J/(kg·K); higher is better)

The registered objective is its negative. Pure values are campaign 11's measured
pure rows, and the excesses come from `../viscosity_excess/export_excess.py`.
Campaigns 1–10 are left out: they were a BO run that steered toward negative
viscosity excess, so their c_p excess does not sample the space. Campaign 12 has no c_p.

**Cleaning.** Raw excess c_p is dominated by single bad measurements. For example, one
row reads −1,186 while the other compositions of the same fluids read −72 to +30.
`build.py` compares each row with the median of its siblings, the other measured
compositions of the same fluids. The robust spread of those differences is
σ = 25 J/(kg·K). Rows more than 5σ from their siblings' median are dropped: 23 of
1,288, leaving 1,265. The outliers are well separated: a 6σ cutoff drops 22 and a
4σ cutoff drops 31. The cutoff `K` is recorded in `provenance.json`.

| | Raw | Cleaned (5σ) |
|---|---|---|
| Nearest-neighbour correlation | 0.08 | 0.43 |
| Fitness-distance correlation | −0.15 | 0.23 |
| Variance explained by the fluid set | 0.45 | 0.83 (4σ) |

- **Features:** the fraction-weighted mean of the components' RDKit 2D descriptors.
  That leaves 109 descriptors (the same scheme as `fluid_mixtures`), and the matrix has
  rank 11.
- **Optimum:** `f_opt = −330` (an excess of +330 J/(kg·K), about 14% above ideal), a
  ternary. Every measured composition of its fluid set has a positive excess, 179 to
  330. The cleaned range is −173 to +330.

**Difficulty.** Most of the signal is which fluids interact, not a smooth trend in
composition. The fraction-weighted descriptors represent that poorly: an ARD GP
fit on 40 random rows ranks the rest with Spearman 0.22. That compares with 0.49 for
`viscosity_excess` and 0.89 for `fluid_mixtures`. This is the hardest of the
three mixture tasks for plain GP-BO.

## Benchmark at budget 50

The baseline is `sobol_gp_bo`: 5 Sobol rows (10% of the budget), then ARD GP-BO with
LogEI (see [`fluid_mixtures`](../fluid_mixtures/README.md#benchmark-at-budget-50)).
Values are the best excess c_p found, in J/(kg·K).

| Run | after 10 | after 20 | after 30 | after 50 | rank of the best found |
|---|---|---|---|---|---|
| `sobol_gp_bo` seed 0 | 33 | 70 | 103 | 103 | 31 |
| seed 1 | 49 | 49 | 99 | 104 | 30 |
| seed 2 | 49 | 49 | 49 | 111 | 28 |
| seed 3 | 247 | 330 | 330 | 330 | 1 (at evaluation 12) |
| seed 4 | 49 | 49 | 49 | 61 | 97 |
| **baseline median** | 49 | 49 | 99 | 104 | 30 |
| agent (Claude Opus, high effort), seed 0 | 87 | 87 | 102 | **163** | 18 |

The agent finished above 4 of the 5 baseline seeds and above the baseline median
(163 against 104). Seed 3 is the exception: it reached 247 within its first 10 rows
and found the optimum at evaluation 12, a lucky start that the GP itself did not
earn. Neither method reached the top 1% reliably.

The agent (run `run-0fb14215dbc9`, audit passed, 6 calls) took these steps:
1. a k-means design of 12 rows on 10 principal components, about 1.2× the effective
   dimension, even under `SKILL.md`'s budget-cost wording;
2. its own GP with LogEI on 5, then 10, principal components (`work/pca_gp_ei.py`),
   resumed from saved state for the last 28 evaluations.

It did not model the fluid-set structure that carries most of the signal. That
remains the obvious headroom.

The data is proprietary and lives in git-ignored `external/fluid_mixtures/`. Only
`build.py`, this README and `provenance.json` are tracked. Rebuild with an RDKit
environment, after exporting `excess_rows.csv` as described in
[`viscosity_excess`](../viscosity_excess/README.md):

```bash
~/Documents/Work/Grad/sdagent/.venv/bin/python data/cp_excess/build.py
.venv/bin/python scripts/run_loop.py --task cp_excess --budget 50 --seed 0
.venv/bin/python scripts/run_campaign.py --task cp_excess --optimizer sobol_gp_bo --seed 0 --budget 50
```
