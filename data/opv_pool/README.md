# OPV device pool task (`opv_pool`)

The 526 OPV devices of Seifrid et al. (2024) as a fixed pool. The search box is
[0, 1]^11 (min–max scaled device features). Each evaluation maps `x` to the
nearest device not yet tested in this campaign (Euclidean distance, ties to the
lowest `candidate_id`) and returns that device's measured −PCE (%), so the same
`x` evaluated twice returns two different devices and a campaign allows at most
526 evaluations. `f_opt` is the best measured device (exact). The task is
`src/agentoptimizer/tasks/opv_pool.py`, registered as `opv_pool`; its results folder name
is `opv_pool_pce_d11`.

This folder names the task's identity. Never copy it into an agent workspace
(the runner never does).

## Files

| File | Role |
| --- | --- |
| `opv_preprocessing_prompt_pool.md` | Specification of the pool |
| `pce_cleaned_dataset.csv` | Input: cleaned device dataset |
| `preprocess.py` | Builds the pool from the dataset and checks it |
| `pool_features.csv`, `pool_labels.csv`, `pool_spec.json` | The pool, read by the task |
| `initial_sets.json`, `preprocessing_report.md` | Other outputs of `preprocess.py`: initial designs (not used by the task) and the checks with every deviation from the specification |

## Run it

Set up both environments and the Codex CLI once, as in the root `README.md`
("Install once"). Run every command from the repository root. `runs/` is
git-ignored; results are released separately, not committed.

1. Optional: rebuild the pool. The committed files are already its output.

   ```bash
   .agent-venv/bin/python data/opv_pool/preprocess.py
   ```

2. Smoke test:

   ```bash
   .venv/bin/python scripts/run_campaign.py --task opv_pool \
     --optimizer random_search --seed 0 --budget 20 --outdir runs/smoke
   ```

3. Baselines at budget 100: every optimizer, including the pool-only
   `pool_random` (uniform picks without replacement). Ten seeds, one for the
   deterministic BOBYQA, Nelder–Mead and L-BFGS-B, so 73 campaigns. The
   published ones, with their plots, are `main/results_full/opv_pool_pce_d11/` in
   the source repository. To
   rerun them (TuRBO traces vary slightly from run to run, on every task):

   ```bash
   for opt in random_search sobol cma_es de gp_bo turbo pool_random; do for s in $(seq 0 9); do
     .venv/bin/python scripts/run_campaign.py --task opv_pool --optimizer $opt --seed $s --budget 100 --outdir runs/baselines-100
   done; done
   for opt in bobyqa nelder_mead lbfgsb; do
     .venv/bin/python scripts/run_campaign.py --task opv_pool --optimizer $opt --seed 0 --budget 100 --outdir runs/baselines-100
   done
   ```

4. Agent run. Check isolation first (zero evaluations, no model call), then
   run. Claude Code is the default agent; add `--agent codex` for Codex.

   ```bash
   .venv/bin/python scripts/run_loop.py --task opv_pool --budget 100 --check-only
   .venv/bin/python scripts/run_loop.py --task opv_pool --budget 100
   ```

   The agent is told only the dimension, the [0, 1] box, the budget and that
   the objective is not repeatable. It never sees the pool or the devices.

Budgets above 526 are refused by `run_campaign.py` and `run_loop.py`.
