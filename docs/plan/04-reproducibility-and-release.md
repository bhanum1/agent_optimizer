# 04 · Reproducibility and release

## Environments

- Done: uv manages both environments from `pyproject.toml` and `uv.lock`
  (Python 3.11; 3.11.9 on the development Mac). `[tool.uv]
  constraint-dependencies` pins the direct libraries to the versions behind
  the published runs (numpy 2.4.6, torch 2.14.1, botorch 0.18.1, cma 4.5.0,
  …). The agent environment is the `agent` dependency group alone and never
  sees the package or objective backends.
- Linux: `uv.lock` is cross-platform, but torch wheels differ by platform;
  check results on the cluster against macOS before mixing them.
- Record the Codex CLI and Claude Code CLI versions, the model identifiers and
  reasoning effort for every agent run.
- Validate the Linux sandbox path (bubblewrap) on the target cluster; the
  seed carries the fixes but the README marks Linux as unchecked.

## Which runs are rerun from this repository

| Experiment | Existing runs (old repo) | Plan |
| --- | --- | --- |
| 30-task suite, 2,000 evaluations | `main/agent_runs/astra_max_2000_20261022` | Decide: rerun from tagged release, or cite old runs with commit hash |
| OPV and MOF pools, 100 evaluations, 3 seeds | `main/agent_runs/materials_astra_max_100_3seeds_20261022` | Same decision |
| Baseline sweeps | `main/results_linux`, `main/results_full` | Rerun with pinned environments if versions change |
| Dynamic selectors (02), LLM comparisons (03) | — | Run from this repository |

## Data release

- Results stay out of git. Candidate host: Zenodo (versioned DOI, can be
  linked to GitHub releases).
- Bundle per experiment: traces, `meta.json`, roadmaps and changelogs, and
  the manifest of code commit, environment and model versions.
- Pre-cutover results come from the old repo at `pre-manuscript-snapshot`.

## License and citation

- Code license: to decide with co-authors.
- Data: check the terms recorded in `main/data/mof_pool/provenance.json` and
  the OPV dataset source before redistributing.
- Add `CITATION.cff` and a contributors list that credits everyone in the old
  repository's history.
