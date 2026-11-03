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

**Phase 3 status: done (2026-11-02).** Five headless `claude -p "/opt-run …"`
sessions (Opus 5.5) on the reaction example, with its simulated lab between
them:

| Block | Decision | Best so far |
| --- | --- | --- |
| 1 | Sobol, 8 points | 48.3 |
| 2 | Batch GP-BO (qLogNEI), 8 | 52.0 |
| 3 | GP-BO in a trust region around the incumbent, 8 | 86.2 |
| 4 | Trust region recentred, 7 + 1 replicate (noise sd 0.4) | 92.8 |
| 5 | Refine, 5 + 1 lower-catalyst probe | 93.0 |

The optimum is about 95. This used 38 of 40 experiments, with one failure
in the tar region, which block 2 probed on purpose. It cost about $1.90 and
took 1 to 1.5 minutes per block. No session needed a human. Block 4 fixed
its own script and logged a `**Revision:**`. Every block's `propose.py`
reproduces its proposals exactly from the data it saw.

What it showed for later phases:

- **Each block wrote its own batch BO** (qLogNEI, trust regions) in about
  40 lines, and block 4 needed two fixes. Phase 2's adapters would shorten
  the scripts, remove that class of error, and keep methods consistent
  between blocks. That case is now stronger than the need for batching,
  which the model handled itself.
- **Inline Python was refused in a headless session** (only `opt_run.py` is
  pre-approved), so the model reread files to inspect data and models. The
  skill now says to use Read and to print diagnostics from `propose.py`. A
  deterministic model diagnostic in the summary (GP lengthscales,
  leave-one-out fit) would save these steps.

**Phase 2 status: done (2026-11-02).** `agentoptimizer.batch` gives
`space_filling`, `random`, `qlognei`, `trust_region`, `cma_es` and `de`, all
of which work in pools, plus `combine`. Each block summary also carries
GP diagnostics. The same five-block headless test, rerun on a fresh copy:

| | Phase 3 (hand-written scripts) | Phase 2 (batch proposers) |
| --- | --- | --- |
| Model calls, 5 blocks | $1.90 | $1.57 |
| Turns per block | 19, 9, 18, 20, 10 | 12, 12, 14, 18, 13 |
| `propose.py` code lines | 10 → 59, growing | 6–7 every block |
| Best yield | 93.0 (block 5) | 92.2 (block 3, after 24 experiments) |

Every block reproduces exactly. Its third run in block 4 was deliberate: it
previewed the trust-region box in `propose.log`, then revised the length
and logged it. One run each cannot rank the two on yield.

**Cost, for planning replay (phase 4).** A block costs $0.27–0.46 on Opus
5.5 at default effort, when run headless on a claude.ai login (the figure
is the CLI's API-price estimate). Writing the roughly 20k-token starting
context to the cache is 42–61% of that (every block is a fresh session);
output is 21–37% and cache reads 19–24%. The levers are a cheaper model or
lower effort, restricting headless runs to the tools the skill uses
(`--tools "Bash,Read,Write,Edit"`), and fewer turns (for example,
`new-block` printing everything the decision needs).

### Phase 4 design: the replay runner

`scripts/run_replay.py` (`blind_runner/replay.py`) runs one task as an
advisor campaign. The runner plays the lab; the advisor never touches the
evaluator.

```text
runs/replay-<random>/
  private/     task, evaluations.sqlite, truth copy of the campaign, per-block CLI logs,
               scored_trace.csv, integrity_audit.json, blocks.csv (turns, tokens, cost)
  workspace/   AGENTS.md (advisor policy + headless procedure), opt_run.py,
               lib/agentoptimizer/{advisor,batch}/, .venv (agent runtime view),
               campaign/  problem.toml (x0…x{d-1}, task bounds, minimize y),
                          data.csv, decisions.md, candidates.csv (indexed pools), blocks/
```

Each block:

1. The runner writes `campaign/data.csv` from its private copy.
2. A fresh headless CLI call (`claude -p` or `codex exec`, the benchmark's
   model and effort) follows the same steps as `/opt-run`: check, summarize,
   new-block, decide, write `propose.py`, run, accept. As in the benchmark,
   its only tools are `run_command` and `write_stdin`, both run in the OS
   sandbox, so it cannot read `private/` or reach the network.
3. The runner checks what came back:
   - done rows are unchanged;
   - the only new rows are block N's pending rows, within the remaining
     budget;
   - `decisions.md` is append-only (journaled as a document, as the
     changelog is).
4. It reruns the block's `propose.py` in the sandbox against the data that
   block saw, and requires the same proposals.
5. It evaluates the rows in order through the private evaluator, labelled
   with the block and method, records the values, and loops until the budget
   is used. A call that accepts no block gets one fresh retry, then the run
   stops and is marked incomplete.

The scored trace has the benchmark's format, so the figure pipeline reads both.

Supported for now:
- continuous tasks;
- indexed pools, as `[pool]` campaigns with candidate ids equal to row
  positions;
- non-indexed pools, as continuous campaigns with the benchmark's "not
  repeatable" note.

Constrained tasks wait for constraints in advisor mode (phase 5). Replay uses
`.agent-venv` like the benchmark: `advisor/` and `batch/` use only relative
imports and libraries from the `agent` group, so they are copied in the way
`optimizers/` is, and `propose.py` finds them through the copy. The
condition flag (no summary) comes after the base runner works.

| Slice | Adds | Tests |
| --- | --- | --- |
| 1 | Advisor and batch code that runs from a copy in the agent runtime; evaluator documents configurable per suite | A copied `propose.py` runs under `.agent-venv`; append-only documents |
| 2 | Replay runner loop with an injected advisor and sandbox: workspace, truth copy, checks, reproduction, evaluation | Scripted advisor over three blocks; tampering with results or decisions is refused |
| 3 | Headless model advisor: `prompts/advisor_headless.md`, CLI commands, per-block logs, tool audit, usage | Command construction; one real replay |
| 4 | Launch and status through `agent_runs.py`; conditions (fixed q, no summary) | Launch smoke test |

**Status (2026-11-03): slices 1–4 done (`4e1af57`…`8c6c80d`), except the
condition flags (fixed q, no summary).** The first live replay (sphere10,
budget 24, Opus 5.5 at max effort) found a runner bug: the sandbox denies
its TMPDIR (`.tmp/`), so reproduction could not read `propose.py`. It now
runs in `work/`. That block alone cost $0.97, with 21 turns, 25k output
tokens and 4.4 minutes; it built a custom low-correlation LHS. The rerun at
medium effort finished cleanly:

| Block | Decision | Rows | Turns | Cost |
| --- | --- | --- | --- | --- |
| 1 | Sobol, one point per dimension | 10 | 6 | $0.07 |
| 2 | qLogNEI 6 + centre probe | 7 | 11 | $0.19 |
| 3 | Trust region on the incumbent | 5 | 8 | $0.16 |
| 4 | Trust region, length halved | 2 | 8 | $0.16 |

24/24 evaluations; every block reproduced in the sandbox; the journal
matches the data. The advisor chose its own block sizes (10, 7, 5, 2), not
the preferred 8. A replay block costs well under an `/opt-run` block
($0.07–0.19 against $0.27–0.46) because the headless call has no Claude
Code system prompt or built-in tools, only the policy and two MCP tools.

**Effort: medium (decided 2026-11-03).** The benchmark agent runs at max
effort. Matching it would cost about 5× per block (about $1 instead of
$0.15–0.20). Replay uses medium for every condition, and `run_replay.py`
defaults to it. The comparison with the benchmark agent therefore differs
in effort as well as in batching and fresh context; the paper says so.

**First C6 check (2026-11-03): MOF pool, budget 100, 3 seeds.** The replay
used Claude Opus 5.5 at medium effort. The benchmark agent runs being
compared are the 2026-10-22 runs in the old repository (GPT-6 Astra, max
effort, same data hash), so the comparison differs in model as well as
effort. Mean ± sd regret (pool maximum minus best raw score):

| Method | Seeds | @25 | @50 | @100 | Found the best |
| --- | --- | --- | --- | --- | --- |
| Benchmark agent (live, sequential) | 3 | 2.17 ± 1.49 | 0.29 ± 0.17 | 0.16 ± 0.28 | 2/3 |
| Advisor replay | 3 | 1.96 ± 2.59 | 1.88 ± 2.64 | 0.25 ± 0.43 | 2/3 |
| GP-BO, sequential | 10 | 3.40 ± 2.07 | 1.89 ± 2.04 | 0.34 ± 0.79 | 7/10 |
| TuRBO, sequential | 10 | 2.87 ± 2.07 | 1.61 ± 1.79 | 0.63 ± 0.90 | 4/10 |
| Sobol → qLogNEI, q=8 | 10 | 3.96 ± 0.00 | 3.44 ± 1.24 | 1.82 ± 1.32 | 2/10 |
| Sobol → trust region, q=8 | 10 | 3.89 ± 0.21 | 3.66 ± 0.39 | 3.01 ± 0.68 | 0/10 |
| Random, q=8 | 10 | 3.49 ± 1.58 | 3.09 ± 1.48 | 2.84 ± 1.34 | 1/10 |

Caveats:
- Replay seed 0 stopped at 92/100 when the account's usage limit was hit;
  its @100 value is its value at 92.
- Each replay block cost about $0.40, against about $0.15 on sphere10,
  because the 193-feature summaries are large. The three runs cost $14.4.
- **Bug found:** `batch.space_filling` ignores its seed on pools. Every
  baseline seed and replay seed 1 drew the same first block, which
  understates the batch baselines' spread. Fix it and rerun the batch
  baselines (no model calls) before reading these numbers as final.

Reading: the advisor ends close to the live agent and ahead of every batch
baseline at the same q. It is slower to start in two of three seeds. Three
seeds show direction, not a difference.

The advisor also wrote methods of its own when the building blocks did not
fit, labelled `nearest_neighbours` and `gp_mean_consensus`: the freedom the
design asks for is already used.

### Block size and context (decided 2026-11-03)

**Block size is the advisor's choice, sized against the budget.**
`[budget] block_size` stays a suggestion, and there is no hard cap per
block. The advisor already sees the total and remaining budget, in
`problem.toml` and in every summary. What the policy lacks is the rule
that a block is sized against them: with 100 evaluations, a block of 50
leaves room for one more decision, and a block of 2 spends the budget on
model calls. `prompts/advisor.md` gains that sentence when the unified
loop's policy is written (06, slice 1). It is a prompt change, and so a
change of condition.

**Context: fresh every block.** Each block stays a new headless session,
and continuity comes only from the campaign folder, mainly `decisions.md`.
This is how replay already works, and nothing changes. The reason is cost.
In the MOF replays, fresh blocks already read the fixed prefix (system
prompt and tools, about 5.4k tokens) from cache across sessions, since
blocks run back to back. Each block's 25–48k written tokens are new
material: the summary, data reads, the script. A resumed session would
save little and would add the whole history to every turn. At about 30k
tokens per block and 11 turns, block 12 of a MOF run would read around 4M
cached tokens, against about 250k fresh. The resume option and the
"fresh vs. resumed" comparison are dropped. The planned "fixed q"
condition is dropped too.

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
- **Shared policy text** (phase 3). `prompts/advisor.md` is its own file
  carrying the benchmark prompt's block-wise policy; the contributor guide
  says to keep the two in step. The benchmark prompt is unchanged.
- **Where the adapters live** (phase 2). A new package,
  `agentoptimizer.batch`, of stateless proposers: each call takes the data
  so far (unit cube, loss, pending, failed, pool candidates) and returns the
  next q points with the settings it used and diagnostics. The same form
  serves the selectors in 02, which can call them with the full history.
  `optimizers/` is unchanged, so the benchmark workspace is too.

## Open decisions

1. **Domain context.** Allowed by default in advisor mode, but replay uses
   anonymous names. A follow-up could test whether real parameter names (OPV
   has them) help.

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
