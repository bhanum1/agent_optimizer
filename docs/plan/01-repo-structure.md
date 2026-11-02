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
  harness/  blind_runner/  optimizers/  selectors/  tasks/  llm_baselines/
scripts/                          run_*, figures/
data/                             task inputs only: mof_pool/, opv_emulator/, opv_pool/
experiments/                      one config per paper experiment (02, 03, 04)
tests/
docs/                             user docs; plan/ removed or archived at release
```

`selectors/` holds the dynamic-selection baselines (02); `llm_baselines/`
holds the LLM comparisons (03). Both are created by the PRs that add the
first method, not ahead of time.

## Open decisions

- **Flatten `main/`.** Nothing else lives beside it now, so the extra level only
  lengthens paths. Moving to `src/agentoptimizer/` with a `pyproject.toml`
  replaces the `sys.path.insert(0, 'main')` pattern used in scripts and docs.
- **Agent prompt location.** `CLAUDE.md` imports `AGENTS.md`, which is the
  *optimization agent's* prompt. Any coding assistant opened on this repo
  therefore reads instructions meant for the experiment agent. Move the prompt
  to `prompts/` and give contributors' assistants their own short `AGENTS.md`.
- **OPV pipelines.** `task_005_baseline/` and `task_006_baseline/` mix task
  inputs (read by `opv.py` and `opv_pool.py` via a hard-coded `parents[3]`
  path) with preprocessing code, notebooks and reports. Keep only inputs under
  `data/`; move the pipelines to `data/<task>/pipeline/` or drop them if the
  data release documents provenance.
- **`psa_v10`.** Registered but needs a private evaluator that is not
  distributed. Drop from the release, or keep with a clear "not reproducible"
  note.

## Cleanup list

- [ ] README links to `main/agent_runs/...` and `task_006_baseline/README.md`
      results that are not in this repository; point them at the data release.
- [ ] README paragraph on the PSA campaign scripts, which were not imported.
- [ ] README layout section mentions `archive_results/` and `poly_results/`.
- [ ] Figure scripts default to result folders (`results_linux`, etc.) that
      will live in the data release; make the input path an argument.
- [ ] Hard-coded data paths in `opv.py` and `opv_pool.py`.
