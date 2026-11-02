# 01 · Repository structure

## Seed layout (as imported)

```
AGENTS.md, CLAUDE.md, README.md
main/
  src/agent_harness/      in-process harness (ask the objective, log trace)
  src/blind_runner/       isolated agent runner, evaluator, preflight checks
  src/optimizers/         ask/tell wrappers, OPTIMIZERS registry
  src/tasks/              task definitions, TASKS registry
  scripts/                run_campaign, run_sweep, run_agent, make_*_plot, slurm/
  tests/                  integrity and pool tests (22 tests, all pass on the seed)
  data/mof_pool/          MOF pool data and provenance
  TASKS.md, ADDING_TASKS.md, FIGURES.md, requirements*.txt
task_005_baseline/        OPV emulator data, model and preprocessing pipeline
task_006_baseline/        OPV pool data and preprocessing pipeline
```

Not imported: `main/results*`, `main/agent_runs`, `archive_results`,
`poly_results`, `video`, `main/task_definitions`, and the historical PSA
campaign scripts. They remain in the old repository at the cutover tag.

## Target layout (proposal)

```
README.md  LICENSE  CITATION.cff  pyproject.toml
prompts/optimization_agent.md     the agent prompt (today's AGENTS.md)
src/agentoptimizer/
  agent_harness/  blind_runner/  optimizers/  selectors/  tasks/  llm_baselines/
scripts/                          run_*, figures/
data/                             task inputs: mof_pool/, opv_emulator/, opv_pool/
experiments/                      one config per paper experiment (02, 03, 04)
tests/
docs/                             user docs; plan/ removed or archived at release
```

`selectors/` holds the dynamic-selection baselines (02); `llm_baselines/`
holds the LLM comparisons (03). Both are created by the PRs that add the
first method, not ahead of time.

## Decisions

Made in the restructure commit:

- **Flattened `main/`** into `src/agentoptimizer/` with a `pyproject.toml`;
  environments come from `uv sync` (see the README; uv replaced the
  `requirements*.txt` files).
  Scripts and tests no longer edit `sys.path` to find the package.
- **Agent-facing layout unchanged.** The runner still copies the harness and
  optimizers into each workspace as package `src`, so the agent sees the same
  files and import paths as in earlier runs. The copied modules now use
  relative imports so they work under both names. The runner and preflight
  also refuse an agent runtime that has `agentoptimizer` installed.
- **Agent prompt moved** to `prompts/optimization_agent.md` (copied into each
  workspace as `AGENTS.md`). The root `AGENTS.md` is now a contributor guide.
- **OPV data**: `task_005_baseline/` → `data/opv_emulator/`,
  `task_006_baseline/` → `data/opv_pool/`, each moved whole so the pipelines
  still run in place. `make_poly_results.py` was dropped; it only plotted
  results that are not in this repository.
- **Figure scripts** moved to `scripts/figures/` and resolve result folders
  from the repository root, where a results release would be unpacked.

## Open decisions

- **`psa_v10`.** Registered but needs a private evaluator that is not
  distributed. Drop from the release, or keep with a clear "not reproducible"
  note.
- **Data pipelines vs. inputs.** `data/opv_emulator/` and `data/opv_pool/` keep
  their preprocessing code, notebook and reports beside the inputs, and both
  carry a copy of `pce_cleaned_dataset.csv`. Trim once the data release
  documents provenance.
- **Figure-script inputs.** Several scripts hard-code result folders
  (`results_linux`, `results_full`, `agent_runs/...`) and the old
  `AGENT_RUNS` map in `make_comparison_plots.py` points at archived runs.
  Rework once the data release layout is fixed.

## Cleanup list

- [x] README links to results that are not in this repository; they now point
      at the source repository's tag.
- [x] README paragraph on the PSA campaign scripts.
- [x] README layout section.
- [ ] Figure scripts: make the input path an argument everywhere.
- [x] Hard-coded data paths in `opv.py` and `opv_pool.py` (now `data/...`).
