# sonnet_medium

Model claude-sonnet-5-5. One folder per task and seed; each has `trace.md` (the agent's reasoning and decisions, call by call), `data.csv` and `scripts/` (its optimizers).

| Task | Seed | Final log10 regret | Calls | Cost | Methods, block by block |
|---|---|---|---|---|---|
| [ellipsoid](ellipsoid/seed_0/trace.md) | 0 | 0.16 | 5 | $0.32 | cma1 → cma1 → cma1 → cma1 → cma1 |
| [ellipsoid](ellipsoid/seed_1/trace.md) | 1 | -3.31 | 4 | $0.27 | ipop → cma_local → cma_local → cma_local |
| [ellipsoid](ellipsoid/seed_2/trace.md) | 2 | -4.31 | 5 | $0.33 | cma → cma → cma → cma2 → loc |
| [gallagher](gallagher/seed_0/trace.md) | 0 | -4.04 | 5 | $0.28 | explore → bo → bo → local1 → local2 |
| [gallagher](gallagher/seed_1/trace.md) | 1 | -5.39 | 4 | $0.23 | bo → bo → bo → local |
| [gallagher](gallagher/seed_2/trace.md) | 2 | 0.10 | 4 | $0.21 | bo → bo → local1 → local2 |
| [noisy_ellipsoid](noisy_ellipsoid/seed_0/trace.md) | 0 | 2.01 | 6 | $0.41 | sobol → bolog → bolog → bolog → lbo → lbo |
| [noisy_ellipsoid](noisy_ellipsoid/seed_1/trace.md) | 1 | -3.88 | 5 | $0.31 | sobol → cma1 → cma1 → cma1 → refine |
| [noisy_ellipsoid](noisy_ellipsoid/seed_2/trace.md) | 2 | -2.23 | 7 | $0.45 | sobol → cma1 → cma1 → cma1 → cma1 → cma2 → polish |
| [noisy_sphere](noisy_sphere/seed_0/trace.md) | 0 | -2.30 | 4 | $0.27 | sobol → quad → refine → fine |
| [noisy_sphere](noisy_sphere/seed_1/trace.md) | 1 | -1.46 | 5 | $0.37 | probe → sepquad → qlocal → qlocal → qlocal |
| [noisy_sphere](noisy_sphere/seed_2/trace.md) | 2 | 0.16 | 4 | $0.22 | sobol → cma → cma → cma |
| [step_ellipsoid](step_ellipsoid/seed_0/trace.md) | 0 | -8.00 | 4 | $0.23 | cma1 → cma1 → cma1 → cma2 |
| [step_ellipsoid](step_ellipsoid/seed_1/trace.md) | 1 | -8.00 | 5 | $0.30 | cma → cma → cma → polish → polish |
| [step_ellipsoid](step_ellipsoid/seed_2/trace.md) | 2 | 0.19 | 5 | $0.40 | sobol → cma → ls → ls → ls_multi |
| [very_noisy_sphere](very_noisy_sphere/seed_0/trace.md) | 0 | -1.46 | 5 | $0.34 | sobol → bo1 → bo1 → focus → focus2 |
| [very_noisy_sphere](very_noisy_sphere/seed_1/trace.md) | 1 | -0.81 | 7 | $0.49 | bo → rep → bo2 → cma1 → cma1 → cma2 → cma1 |
| [very_noisy_sphere](very_noisy_sphere/seed_2/trace.md) | 2 | -3.56 | 9 | $0.64 | sobol → bo → rep → elite → elite2 → elite3 → basins → elite4 → elite5 |

Total $6.07 over 18 campaigns.
