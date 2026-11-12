# opus_high

Model claude-opus-5-5. One folder per task and seed; each has `trace.md` (the agent's reasoning and decisions, call by call), `data.csv` and `scripts/` (its optimizers).

| Task | Seed | Final log10 regret | Calls | Cost | Methods, block by block |
|---|---|---|---|---|---|
| [few_evals_10d](few_evals_10d/seed_0/trace.md) | 0 | -8.00 | 8 | $1.67 | sobol_init → quad_probe → local_axial → axial_01 → axref → axref → final_local |
| [few_evals_10d](few_evals_10d/seed_1/trace.md) | 1 | -8.00 | 8 | $1.90 | init_sobol → quad_probe → lq → rep_best → axis_fd → newton → explore_maximin |
| [few_evals_10d](few_evals_10d/seed_2/trace.md) | 2 | -8.00 | 12 | $2.35 | init_sobol → gpbo → quadref → stencil → explore → fine_stencil → newton7 → newton8 → newton9 → newton10 → grid_center |
| [few_evals_10d](few_evals_10d/seed_3/trace.md) | 3 | -8.00 | 11 | $2.20 | sobol_init → quad_center → quad_refine → quad_fine → quad_h02 → quad_h005 → quad_h001 → quad_h0002 → quad_h00005 → final_opt |
| [few_evals_10d](few_evals_10d/seed_4/trace.md) | 4 | -8.00 | 12 | $2.55 | init_sobol → turbo → turbo2 → quad → quadfine → qmin → lottery → fdprobe → newton1 → fdprobe2 → newton2 |
| [few_evals_10d](few_evals_10d/seed_5/trace.md) | 5 | -8.00 | 8 | $1.95 | init_sobol → gpbo_tr → quad_refine → noise_reps → fd_newton → fd_newton_fine → final_reps |
| [ill_conditioned](ill_conditioned/seed_0/trace.md) | 0 | -8.00 | 7 | $2.03 | sobol_init → cma1 → cma1 → tqr1 → tqr2 → tqr3 → tqr4 |
| [ill_conditioned](ill_conditioned/seed_1/trace.md) | 1 | 0.98 | 8 | $1.63 | init_sobol → cma1 → cma1 → bob1 → bob1 → bob2 |
| [ill_conditioned](ill_conditioned/seed_2/trace.md) | 2 | -1.67 | 7 | $1.89 | init_sobol → cma1 → cma1 → boby1 → cmaloc → cmaloc → cmafin |
| [ill_conditioned](ill_conditioned/seed_3/trace.md) | 3 | -6.76 | 9 | $2.93 | sobol_init → cma → cma → trq → trq → trq2 → trq3 → trq3 → trq3 |
| [ill_conditioned](ill_conditioned/seed_4/trace.md) | 4 | 0.41 | 11 | $2.72 | sobol_init → bobyqa1 → bobyqa1 → floor1 → bobyqa1 → bobyqa2 → tlq1 → tlq1 → tlq2 |
| [ill_conditioned](ill_conditioned/seed_5/trace.md) | 5 | 1.35 | 12 | $2.29 | sobol_init → turbo → cma → cma_glob → cma_b → bobyqa → bobyqa → bobyqa_fin |
| [ill_conditioned_noisy](ill_conditioned_noisy/seed_0/trace.md) | 0 | -2.58 | 9 | $1.63 | init_sobol → cma1 → noise_rep → cma1 → cma1 → cma2 → cma2 → cma2 → cma2 |
| [ill_conditioned_noisy](ill_conditioned_noisy/seed_1/trace.md) | 1 | -5.48 | 8 | $1.76 | init_sobol → turbo → cma_a → cma_a → cma_a → cma_a → cma_a → quad_final |
| [ill_conditioned_noisy](ill_conditioned_noisy/seed_2/trace.md) | 2 | -8.00 | 14 | $3.14 | sobol_init → noise_probe → cma1 → cma1 → noise_probe2 → bq1 → cmaw1 → cmax1 → cmax1 → bw10 → explore11 → final_reps |
| [ill_conditioned_noisy](ill_conditioned_noisy/seed_3/trace.md) | 3 | -0.28 | 13 | $2.67 | init_sobol → turbo1 → reps1 → facecma1 → cma5a → cma5a → cma5a → cma5a → cma5a → cma5a → cma5a |
| [ill_conditioned_noisy](ill_conditioned_noisy/seed_4/trace.md) | 4 | -2.50 | 16 | $3.28 | init_sobol → turbo1 → turbo1 → rep1 → cma1 → cma1 → cma2 → cma2 → cma2 → trprobe → qtr1 → qtr1 |
| [ill_conditioned_noisy](ill_conditioned_noisy/seed_5/trace.md) | 5 | -6.10 | 12 | $2.45 | init_sobol → cma1 → cma1 → cma1 → cma2 → cma2 → cma1 → refine → refine2 → final |
| [noise_dominated](noise_dominated/seed_0/trace.md) | 0 | -4.52 | 14 | $3.77 | sobol_init → turbo1 → turbom → reps_a → rturbo_a → focus_a → focus2_a → endgame_a → shift_a → final_a |
| [noise_dominated](noise_dominated/seed_1/trace.md) | 1 | -3.58 | 11 | $1.75 | reps → turbo2 → turbo2 → turbo3 → reps2 → turbo4 → turbo4 |
| [noise_dominated](noise_dominated/seed_2/trace.md) | 2 | -2.45 | 13 | $1.81 | init_sobol → noise_reps → robo_bo → cma1 → cma1 → cma2 → lbo1 → floor1 → floor2 |
| [noise_dominated](noise_dominated/seed_3/trace.md) | 3 | -5.08 | 12 | $2.09 | sobol_init → noise_reps → gpts → mtr → vts → vts2 → vts3 → endgame |
| [noise_dominated](noise_dominated/seed_4/trace.md) | 4 | -2.80 | 14 | $2.78 | sobol_init → reps1 → lbo1 → lbo1 → reps2 → ghedge → probe_near → qtr1 → qtr2 → qtr2 |
| [noise_dominated](noise_dominated/seed_5/trace.md) | 5 | -2.60 | 15 | $2.26 | init_sobol → gpbo → reps1 → gpbo_w → gpbo_w25 → explore_lcb → reps2 → gpbo_w12 → gpbo_w06 → gauss1 → gauss2 |
| [plateaus](plateaus/seed_0/trace.md) | 0 | -8.00 | 11 | $1.86 | init_sobol → turbo → cma → cma_local → gp_explore → cma_fine → cma_fine |
| [plateaus](plateaus/seed_1/trace.md) | 1 | -8.00 | 11 | $1.66 | turbo → cma → cma2 → cmaglob → cmaloc → cmafin |
| [plateaus](plateaus/seed_2/trace.md) | 2 | -2.23 | 13 | $2.28 | sobol_init → turbo → cmar → line_ab → cmar2 → cmaell → turbo_global → cmaell_polish → qtr_final |
| [plateaus](plateaus/seed_3/trace.md) | 3 | -8.00 | 10 | $1.69 | sobol_init → turbo1 → cma1 → cma1 → cma2 → quadtr1 → explore1 |
| [plateaus](plateaus/seed_4/trace.md) | 4 | -8.00 | 6 | $1.30 | init_sobol → turbo1 → ipopcma1 → mscma1 → ipopcma1 |
| [plateaus](plateaus/seed_5/trace.md) | 5 | -0.80 | 10 | $2.37 | sobol_init → turbo → basin → multibasin → multibasin → focus3 → final5 → final2 |
| [unstructured_2d](unstructured_2d/seed_0/trace.md) | 0 | -6.13 | 5 | $0.83 | init_lhs → gp_ei → gap_fill → gp_tr → quad_tr |
| [unstructured_2d](unstructured_2d/seed_1/trace.md) | 1 | 0.20 | 5 | $0.76 | init_sobol → gp_ei → gp_ei → local_b → local_b |
| [unstructured_2d](unstructured_2d/seed_2/trace.md) | 2 | -0.02 | 5 | $0.75 | init_lhs → gpbo → gpbo → trgp → trgp |
| [unstructured_2d](unstructured_2d/seed_3/trace.md) | 3 | -8.00 | 4 | $0.88 | gpbo → explore → localtr → gapfill |
| [unstructured_2d](unstructured_2d/seed_4/trace.md) | 4 | 0.32 | 4 | $0.71 | gpbo → gpbo_min → gptr → gptr_fine |

Total $70.58 over 35 campaigns.
