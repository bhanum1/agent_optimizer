# opus_high

Model claude-opus-5-5. One folder per task and seed; each has `trace.md` (the agent's reasoning and decisions, call by call), `data.csv` and `scripts/` (its optimizers).

| Task | Seed | Final log10 regret | Calls | Cost | Methods, block by block |
|---|---|---|---|---|---|
| [ellipsoid](ellipsoid/seed_0/trace.md) | 0 | -8.00 | 7 | $2.03 | sobol_init → cma1 → cma1 → tqr1 → tqr2 → tqr3 → tqr4 |
| [ellipsoid](ellipsoid/seed_1/trace.md) | 1 | 0.98 | 8 | $1.63 | init_sobol → cma1 → cma1 → bob1 → bob1 → bob2 |
| [ellipsoid](ellipsoid/seed_2/trace.md) | 2 | -1.67 | 7 | $1.89 | init_sobol → cma1 → cma1 → boby1 → cmaloc → cmaloc → cmafin |
| [ellipsoid](ellipsoid/seed_3/trace.md) | 3 | -6.76 | 9 | $2.93 | sobol_init → cma → cma → trq → trq → trq2 → trq3 → trq3 → trq3 |
| [ellipsoid](ellipsoid/seed_4/trace.md) | 4 | 0.41 | 11 | $2.72 | sobol_init → bobyqa1 → bobyqa1 → floor1 → bobyqa1 → bobyqa2 → tlq1 → tlq1 → tlq2 |
| [ellipsoid](ellipsoid/seed_5/trace.md) | 5 | 1.35 | 12 | $2.29 | sobol_init → turbo → cma → cma_glob → cma_b → bobyqa → bobyqa → bobyqa_fin |
| [gallagher](gallagher/seed_0/trace.md) | 0 | -6.13 | 5 | $0.83 | init_lhs → gp_ei → gap_fill → gp_tr → quad_tr |
| [gallagher](gallagher/seed_1/trace.md) | 1 | 0.20 | 5 | $0.76 | init_sobol → gp_ei → gp_ei → local_b → local_b |
| [gallagher](gallagher/seed_2/trace.md) | 2 | -0.02 | 5 | $0.75 | init_lhs → gpbo → gpbo → trgp → trgp |
| [gallagher](gallagher/seed_3/trace.md) | 3 | -8.00 | 4 | $0.88 | gpbo → explore → localtr → gapfill |
| [gallagher](gallagher/seed_4/trace.md) | 4 | 0.32 | 4 | $0.71 | gpbo → gpbo_min → gptr → gptr_fine |
| [noisy_ellipsoid](noisy_ellipsoid/seed_0/trace.md) | 0 | -2.58 | 9 | $1.63 | init_sobol → cma1 → noise_rep → cma1 → cma1 → cma2 → cma2 → cma2 → cma2 |
| [noisy_ellipsoid](noisy_ellipsoid/seed_1/trace.md) | 1 | -5.48 | 8 | $1.76 | init_sobol → turbo → cma_a → cma_a → cma_a → cma_a → cma_a → quad_final |
| [noisy_ellipsoid](noisy_ellipsoid/seed_2/trace.md) | 2 | -8.00 | 14 | $3.14 | sobol_init → noise_probe → cma1 → cma1 → noise_probe2 → bq1 → cmaw1 → cmax1 → cmax1 → bw10 → explore11 → final_reps |
| [noisy_ellipsoid](noisy_ellipsoid/seed_3/trace.md) | 3 | -0.28 | 13 | $2.67 | init_sobol → turbo1 → reps1 → facecma1 → cma5a → cma5a → cma5a → cma5a → cma5a → cma5a → cma5a |
| [noisy_ellipsoid](noisy_ellipsoid/seed_4/trace.md) | 4 | -2.50 | 16 | $3.28 | init_sobol → turbo1 → turbo1 → rep1 → cma1 → cma1 → cma2 → cma2 → cma2 → trprobe → qtr1 → qtr1 |
| [noisy_ellipsoid](noisy_ellipsoid/seed_5/trace.md) | 5 | -6.10 | 12 | $2.45 | init_sobol → cma1 → cma1 → cma1 → cma2 → cma2 → cma1 → refine → refine2 → final |
| [noisy_sphere](noisy_sphere/seed_0/trace.md) | 0 | -8.00 | 8 | $1.67 | sobol_init → quad_probe → local_axial → axial_01 → axref → axref → final_local |
| [noisy_sphere](noisy_sphere/seed_1/trace.md) | 1 | -8.00 | 8 | $1.90 | init_sobol → quad_probe → lq → rep_best → axis_fd → newton → explore_maximin |
| [noisy_sphere](noisy_sphere/seed_2/trace.md) | 2 | -8.00 | 12 | $2.35 | init_sobol → gpbo → quadref → stencil → explore → fine_stencil → newton7 → newton8 → newton9 → newton10 → grid_center |
| [noisy_sphere](noisy_sphere/seed_3/trace.md) | 3 | -8.00 | 11 | $2.20 | sobol_init → quad_center → quad_refine → quad_fine → quad_h02 → quad_h005 → quad_h001 → quad_h0002 → quad_h00005 → final_opt |
| [noisy_sphere](noisy_sphere/seed_4/trace.md) | 4 | -8.00 | 12 | $2.55 | init_sobol → turbo → turbo2 → quad → quadfine → qmin → lottery → fdprobe → newton1 → fdprobe2 → newton2 |
| [noisy_sphere](noisy_sphere/seed_5/trace.md) | 5 | -8.00 | 8 | $1.95 | init_sobol → gpbo_tr → quad_refine → noise_reps → fd_newton → fd_newton_fine → final_reps |
| [step_ellipsoid](step_ellipsoid/seed_0/trace.md) | 0 | -8.00 | 11 | $1.86 | init_sobol → turbo → cma → cma_local → gp_explore → cma_fine → cma_fine |
| [step_ellipsoid](step_ellipsoid/seed_1/trace.md) | 1 | -8.00 | 11 | $1.66 | turbo → cma → cma2 → cmaglob → cmaloc → cmafin |
| [step_ellipsoid](step_ellipsoid/seed_2/trace.md) | 2 | -2.23 | 13 | $2.28 | sobol_init → turbo → cmar → line_ab → cmar2 → cmaell → turbo_global → cmaell_polish → qtr_final |
| [step_ellipsoid](step_ellipsoid/seed_3/trace.md) | 3 | -8.00 | 10 | $1.69 | sobol_init → turbo1 → cma1 → cma1 → cma2 → quadtr1 → explore1 |
| [step_ellipsoid](step_ellipsoid/seed_4/trace.md) | 4 | -8.00 | 6 | $1.30 | init_sobol → turbo1 → ipopcma1 → mscma1 → ipopcma1 |
| [step_ellipsoid](step_ellipsoid/seed_5/trace.md) | 5 | -0.80 | 10 | $2.37 | sobol_init → turbo → basin → multibasin → multibasin → focus3 → final5 → final2 |
| [very_noisy_sphere](very_noisy_sphere/seed_0/trace.md) | 0 | -4.52 | 14 | $3.77 | sobol_init → turbo1 → turbom → reps_a → rturbo_a → focus_a → focus2_a → endgame_a → shift_a → final_a |
| [very_noisy_sphere](very_noisy_sphere/seed_1/trace.md) | 1 | -3.58 | 11 | $1.75 | reps → turbo2 → turbo2 → turbo3 → reps2 → turbo4 → turbo4 |
| [very_noisy_sphere](very_noisy_sphere/seed_2/trace.md) | 2 | -2.45 | 13 | $1.81 | init_sobol → noise_reps → robo_bo → cma1 → cma1 → cma2 → lbo1 → floor1 → floor2 |
| [very_noisy_sphere](very_noisy_sphere/seed_3/trace.md) | 3 | -5.08 | 12 | $2.09 | sobol_init → noise_reps → gpts → mtr → vts → vts2 → vts3 → endgame |
| [very_noisy_sphere](very_noisy_sphere/seed_4/trace.md) | 4 | -2.80 | 14 | $2.78 | sobol_init → reps1 → lbo1 → lbo1 → reps2 → ghedge → probe_near → qtr1 → qtr2 → qtr2 |
| [very_noisy_sphere](very_noisy_sphere/seed_5/trace.md) | 5 | -2.60 | 15 | $2.26 | init_sobol → gpbo → reps1 → gpbo_w → gpbo_w25 → explore_lcb → reps2 → gpbo_w12 → gpbo_w06 → gauss1 → gauss2 |

Total $70.58 over 35 campaigns.
