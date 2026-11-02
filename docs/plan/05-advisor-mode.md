# 05 · Advisor mode

The repository so far has one mode, built to **measure** the agent: a blind,
sandboxed campaign in which the agent calls a private objective and the
evaluator enforces the budget. Advisor mode is for **using** it. The user owns
the objective (a lab, a simulation queue) and a folder of results so far. One
skill call reads that data and the decisions made so far, chooses the method,
settings and size of the next block, and returns a Python script that
proposes the block. The user runs the experiments, records the results and
calls again.

Supports a possible claim C6: the agent's block-wise policy still works when
each block is decided from a fresh context and proposed as a batch, which is
how real experimental campaigns run. Phase 4 measures it.

## How the two modes differ

| | Benchmark mode (`/optimize`) | Advisor mode (`/opt-run`) |
| --- | --- | --- |
| Who evaluates | Private evaluator, through the bridge | The user, outside the loop (lab, cluster, simulator) |
| What the model sees | Bounds, dimension, budget, its own observations; no task identity | Whatever the user provides: parameter names, units, notes, data from any source |
| Isolation | OS sandbox, preflight, integrity audit | None needed for blindness; the user reviews and runs the script |
| Continuity between blocks | One live session; optimizer processes kept alive with `write_stdin` | A fresh call per block; continuity only through the campaign folder |
| Block shape | Sequential: one evaluation at a time, can stop early | Batch: q points proposed up front, pending until measured |
| Optimizer state | In memory | Rebuilt from the data at each block |
| Budget | Enforced by the evaluator | Declared by the user; advisory |
| Output | `trace.csv`, roadmap, changelog | Decision entry, `propose.py`, `proposals.csv` |

The decision policy should stay the same in both modes: decide one bounded
block, choose its length, reconsider hyperparameters, record why, and plan no
further than the next block. That way the benchmark results say something
about the advisor.

## Campaign folder (the interface)

```text
my_campaign/
  problem.toml      parameters (name, type, bounds or levels, unit), objective name
                    and direction, constraints, noise, total budget, preferred
                    block size, free-text context
  data.csv          one row per experiment: id, block, optimizer label, parameter
                    columns, objective, constraint columns, status
                    (done | pending | failed), source (advisor | manual | import), time
  decisions.md      append-only, one entry per block; the benchmark changelog's fields
  candidates.csv    optional: a finite pool to choose from (as in mof_pool)
  blocks/003/
    summary.json    the data summary the advisor was given
    propose.py      the script it wrote
    proposals.csv   what the script produced
```

A folder rather than a database: people can read it, diff it, and keep it in
their own version control or lab notebook. TOML for the problem file because
Python 3.11 reads it without a new dependency (`tomllib`).

## The proposal script

`propose.py` is the deliverable of each call. Its contract:

- Reads `problem.toml`, `data.csv` and `candidates.csv` if present. Writes
  exactly q rows to `proposals.csv`, in the user's units, and nothing else.
- Opens with a docstring that repeats the decision: method, hyperparameters,
  block size, seed and reason.
- Deterministic: a fixed seed, and it records the package versions and commit;
  rerunning it reproduces the same proposals.
- Uses only the evaluator environment (`.venv`, pinned by `uv.lock`) and
  `agentoptimizer`; no network.

```python
"""Block 3: TuRBO, batch of 8, trust region restarted at the incumbent.
Why: blocks 1-2 (Sobol, GP-BO) found one basin; ... (as in decisions.md)"""
from agentoptimizer.advisor import Campaign, methods

campaign = Campaign.load(__file__)
X, y = campaign.observations(scaled=True)        # done rows, in [0, 1]^d
turbo = methods.TuRBO(campaign.unit_bounds, seed=303, batch_size=8)
turbo.warm_start(X, y, pending=campaign.pending(scaled=True))
campaign.write_proposals(turbo.ask(8), label="turbo")
```

Later (phase 5), `propose.py --evaluate module:function` runs the block in
closed loop for computational objectives and appends the results.

## Warm start and batches

Advisor mode needs two things the current wrappers lack. They must start from
a history they did not produce, and they must propose q points at once while
other points are still pending.

| Method | Start from any history | Batch of q | Plan |
| --- | --- | --- | --- |
| Sobol, random | Yes | Yes | Skip the Sobol points already used (fast-forward by the count) |
| GP-BO | Yes (`tell` accepts any data) | No (`ask(1)` only) | q-LogNEI, with pending points passed as `X_pending` |
| TuRBO | GP yes; trust-region length depends on its success history | Yes (batch fixed at 4) | Batch size as a setting; length restarted at the incumbent, or rebuilt by replaying blocks in order |
| CMA-ES | No (`tell` must match its own `ask`) | Population = batch | Mean at the weighted best μ points, σ from their spread; optionally covariance from them |
| DE | No (same) | Generation = batch | Initial population from the best NP points |
| BOBYQA, Nelder–Mead, L-BFGS-B | Sequential | No | Offered only for q = 1 or in closed loop |
| Pool GP-BO, pool TuRBO | Yes | Yes, with exclusion | Exclude done and pending rows |

These adapters go in `src/agentoptimizer/advisor/methods.py`, not in
`optimizers/`. The runner copies `optimizers/` into every benchmark workspace,
so changing it changes an experimental condition (see AGENTS.md). Plan 02's
selectors need the same warm start; they should share these adapters.

By default, state is rebuilt from the data at every block: that is portable
and auditable. Saved optimizer state (a pickle keyed by package versions) is
only a cache.

## What the model reads

`opt_run.py summarize` computes a fixed summary instead of leaving the model to
work out statistics from a raw table:
- counts of done, pending and failed rows, and budget used;
- the incumbent and its parameters;
- per block: method, size, best value and improvement over the previous incumbent;
- recent improvement rate;
- how far the last block's points were from the incumbent;
- the share of points on a bound, and near-duplicates;
- noise from replicates if there are any;
- the pattern of failed experiments.

The summary is saved with the block, so each decision can be traced to what
the model saw. The raw table stays available.

## The `/opt-run` skill

1. `/opt-run init` builds `problem.toml` from a short conversation and imports
   existing results (CSV or Excel) into `data.csv`. `/opt-run <folder>` runs
   the steps below.
2. `opt_run.py check`: validates the schema, types, bounds and units. It stops
   with a clear error rather than guessing.
3. `opt_run.py summarize` writes `blocks/NNN/summary.json`.
4. Decide: the model reads `problem.toml`, the summary, `decisions.md` and
   the last block's script, following `prompts/advisor.md`. It chooses the
   method, hyperparameters, block size, seed and reason.
5. It writes `propose.py` and runs it with `.venv/bin/python`. Then
   `opt_run.py validate` checks the count, bounds and types, rejects duplicates
   of done or pending rows, and checks pool membership.
6. It appends the decision to `decisions.md`, adds the proposals to
   `data.csv` as pending rows, and shows the table.
7. The user runs the experiments, enters the results (or runs
   `opt_run.py record results.csv`) and calls `/opt-run` again. Points the user
   overrides or adds are recorded with `source = manual`, and the next
   decision sees them.

## Validation: advisor replay (C6)

Run advisor mode inside the blind benchmark. The runner puts a campaign
folder with anonymous parameter names (`x0` … `x{d-1}`) in the workspace. At
each block a fresh headless advisor call writes `propose.py`. The runner runs
it in the sandbox, evaluates the proposals through the private evaluator and
appends the results, until the budget is used. Same tasks, budgets and seeds
as C1 and C4.

| Comparison | Isolates |
| --- | --- |
| Benchmark agent (live session, sequential) | Combined cost of batching and fresh context |
| Advisor with fixed q vs. advisor-chosen q | Value of choosing the block size |
| Batch baselines at the same q (q-LogNEI, TuRBO, CMA-ES) and SBS/VBS | Value of adaptive selection in batch mode |
| With and without the summary | Whether the summary helps or steers |

Start with the OPV and MOF pools (100 evaluations, closest to a lab campaign),
then a subset of the 30-task suite. Report model calls and tokens per block
alongside the scores.

## Phases

| Phase | Deliverable | Model calls | Depends on |
| --- | --- | --- | --- |
| 0 | Settle the open decisions below | — | — |
| 1 | Campaign format; `agentoptimizer.advisor` (load, check, summarize, validate, record); `scripts/opt_run.py`; tests | None | 0 |
| 2 | Warm-start and batch adapters for the table above; tests on synthetic functions | None | 1; shared with 02 |
| 3 | `prompts/advisor.md`; the `/opt-run` skill (in-session); an example campaign in `examples/` | Small | 1, 2 |
| 4 | Headless advisor (`agent codex\|claude`); blind replay runner; experiments | Yes | 3; Linux check (04) for the cluster |
| 5 | Categorical variables, closed-loop `--evaluate`, constraints, multi-objective, multi-fidelity | — | 4 |

### Phase 1 in slices

Each slice is one layer with its own tests and commit, and the next slice
builds on it. Every slice is usable from `scripts/opt_run.py` as soon as it
lands.

**Status: done (2026-11-02, `bddd132`…`03c8edb`, 36 tests).** Left for later
phases, found while building it:

- Batches are still one point per call for GP-BO and pool GP-BO; phase 2's
  batch adapters are what make real blocks possible.
- Proposals are rounded to 6 significant digits. A per-parameter `step`
  (for example 5 °C increments) would match lab practice better.
- Import reads CSV only, not Excel.
- `accept` requires a `## Block N` entry in `decisions.md` but does not
  enforce append-only edits, as the benchmark evaluator does.

| Slice | Adds | CLI | Tests |
| --- | --- | --- | --- |
| 1 | `problem.toml` schema and loader; scaling to the unit cube (linear, log, integer) | `init`, `check` | Valid and invalid specs; scaling round trip |
| 2 | `data.csv` schema; reader and writer that keep extra columns; row checks | `check` | Each error names its line and column |
| 3 | Deterministic summary | `summarize` | Incumbent, block table and distances on hand-made data |
| 4 | Block folders; proposal validation; accepting proposals as pending rows; running `propose.py` | `new-block`, `run`, `validate`, `accept` | Rejects out-of-bounds points, duplicates and a missing decision |
| 5 | Results in: pending rows recorded, prior data imported, manual rows | `record`, `import` | Status changes; unknown ids refused; all-or-nothing writes |
| 6 | `Campaign` API for proposal scripts; a loop over three blocks with no model | — | End to end on a synthetic function |
| 7 | Candidate pools (`candidates.csv`) through every layer | all | Proposals must be untested pool rows |

Phase 3 is done when a synthetic campaign goes through five blocks of
`/opt-run` → proposals → simulated results without manual fixes, and every
block's `propose.py` reproduces its proposals exactly when rerun.

## Decisions made (2026-11-02)

- **Name.** The skill is `/opt-run` for now; the CLI is `scripts/opt_run.py`
  and the package `agentoptimizer.advisor`.
- **Rebuild freely.** Any part of the repository may be torn down and
  rebuilt; prefer the clean design over compatibility with earlier code or
  runs. Changes to experimental conditions are still recorded, because they
  trigger reruns (04).
- **Who decides.** The calling session in phase 3; a headless `codex exec`
  or `claude -p` call (the benchmarked agent, with fixed model, effort and
  prompt) in phase 4, which replay needs. Both use the same prompt and folder
  contract.
- **Variable types.** Continuous (linear or log scale), integer, and finite
  candidate pools first; categorical in phase 5.
- **Problem file.** TOML, read with the standard library.

## Open decisions

1. **Shared policy text.** Duplicate the block-wise policy of
   `prompts/optimization_agent.md` in `prompts/advisor.md`, or split it into a
   shared file that both prompts include. Splitting changes the benchmark
   prompt file, so the generated `AGENTS.md` must stay byte-identical.
2. **Domain context.** Allowed by default in advisor mode, but replay uses
   anonymous names. A follow-up could test whether real parameter names (OPV
   has them) help.
3. **Where the adapters live** (phase 2). With rebuilding allowed, the
   choice is design, not compatibility: `advisor/methods.py`, or warm start and
   batch support added to `optimizers/` itself and shared with the selectors
   (02). Changing `optimizers/` changes the benchmark workspace, so it means
   reruns.

## Risks

- Batching is a real change of conditions. The benchmark agent can stop a
  block after any single evaluation; the advisor cannot. Replay measures what
  this costs.
- A fresh context per block loses reasoning that is not written down.
  `decisions.md` has to carry what the last block was meant to show and how
  to judge it.
- Hand-edited `data.csv` files will be messy. `check` must be strict and say
  exactly what is wrong.
- `propose.py` is generated code run on the user's machine. The contract
  limits what it may write, `validate` checks its output, and the skill shows
  the script before running it.
