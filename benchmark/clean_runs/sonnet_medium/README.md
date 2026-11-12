# sonnet_medium

Model claude-sonnet-5-5. One folder per task and seed; each has `trace.md` (the agent's reasoning and decisions, call by call), `data.csv` and `scripts/` (its optimizers).

| Task | Seed | Final log10 regret | Calls | Cost | Methods, block by block |
|---|---|---|---|---|---|
| [few_evals_10d](few_evals_10d/seed_0/trace.md) | 0 | -2.30 | 4 | $0.27 | sobol → quad → refine → fine |
| [few_evals_10d](few_evals_10d/seed_1/trace.md) | 1 | -1.46 | 5 | $0.37 | probe → sepquad → qlocal → qlocal → qlocal |
| [few_evals_10d](few_evals_10d/seed_2/trace.md) | 2 | 0.16 | 4 | $0.22 | sobol → cma → cma → cma |
| [ill_conditioned](ill_conditioned/seed_0/trace.md) | 0 | 0.16 | 5 | $0.32 | cma1 → cma1 → cma1 → cma1 → cma1 |
| [ill_conditioned](ill_conditioned/seed_1/trace.md) | 1 | -3.31 | 4 | $0.27 | ipop → cma_local → cma_local → cma_local |
| [ill_conditioned](ill_conditioned/seed_2/trace.md) | 2 | -4.31 | 5 | $0.33 | cma → cma → cma → cma2 → loc |
| [ill_conditioned_noisy](ill_conditioned_noisy/seed_0/trace.md) | 0 | 2.01 | 6 | $0.41 | sobol → bolog → bolog → bolog → lbo → lbo |
| [ill_conditioned_noisy](ill_conditioned_noisy/seed_1/trace.md) | 1 | -3.88 | 5 | $0.31 | sobol → cma1 → cma1 → cma1 → refine |
| [ill_conditioned_noisy](ill_conditioned_noisy/seed_2/trace.md) | 2 | -2.23 | 7 | $0.45 | sobol → cma1 → cma1 → cma1 → cma1 → cma2 → polish |
| [noise_dominated](noise_dominated/seed_0/trace.md) | 0 | -1.46 | 5 | $0.34 | sobol → bo1 → bo1 → focus → focus2 |
| [noise_dominated](noise_dominated/seed_1/trace.md) | 1 | -0.81 | 7 | $0.49 | bo → rep → bo2 → cma1 → cma1 → cma2 → cma1 |
| [noise_dominated](noise_dominated/seed_2/trace.md) | 2 | -3.56 | 9 | $0.64 | sobol → bo → rep → elite → elite2 → elite3 → basins → elite4 → elite5 |
| [plateaus](plateaus/seed_0/trace.md) | 0 | -8.00 | 4 | $0.23 | cma1 → cma1 → cma1 → cma2 |
| [plateaus](plateaus/seed_1/trace.md) | 1 | -8.00 | 5 | $0.30 | cma → cma → cma → polish → polish |
| [plateaus](plateaus/seed_2/trace.md) | 2 | 0.19 | 5 | $0.40 | sobol → cma → ls → ls → ls_multi |
| [unstructured_2d](unstructured_2d/seed_0/trace.md) | 0 | -4.04 | 5 | $0.28 | explore → bo → bo → local1 → local2 |
| [unstructured_2d](unstructured_2d/seed_1/trace.md) | 1 | -5.39 | 4 | $0.23 | bo → bo → bo → local |
| [unstructured_2d](unstructured_2d/seed_2/trace.md) | 2 | 0.10 | 4 | $0.21 | bo → bo → local1 → local2 |

Total $6.07 over 18 campaigns.
