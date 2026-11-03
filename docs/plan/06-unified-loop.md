# 06 · One loop for benchmarks and real campaigns

The repository has two systems built on the same policy:

- **Benchmark mode** (`run_agent.py`, `/optimize`): one live session drives
  `AgentTaskHarness`. It evaluates points one at a time through the bridge
  and keeps optimizer processes alive with `write_stdin`.
- **Advisor mode** (`opt_run.py`, `/opt-run`, replay): each block, the agent
  reads a campaign folder and writes `propose.py`. That script fits the data
  and proposes the block, which a lab (or the replay runner) evaluates.

The second is the design we want everywhere (decided 2026-11-03). The agent
receives data, as a summary or raw. It writes the optimizer for the next
block from building blocks, or from scratch. The optimizer fits the data and
proposes points. Acquisition measures them, and the loop repeats. A benchmark
is this loop with a simulated lab. This document plans the merge. Benchmark
mode then retires; its published runs become a legacy condition.

## The loop

```text
            ┌───────────────────────────────────────────────┐
            ▼                                               │
  campaign/ (problem.toml, data.csv, decisions.md, blocks/) │
            │                                               │
  Advisor   reads the summary or raw data, decides the block,
            writes blocks/N/propose.py          a fresh session every block (05)
            │
  Executor  runs propose.py in the sandbox, checks and reproduces it,
            returns the block's points            agentoptimizer.batch + any code
            │
  Lab       measures the points and records them in data.csv
            simulated: private Evaluator + task (benchmark, replay)
            real:      people, instruments, a cluster queue
            └───────────────────────────────────────────────┘
```

Three parts, each replaceable on its own:

| Part | Interface | Implementations |
| --- | --- | --- |
| Advisor | Campaign folder in; one accepted block (pending rows, a decision entry, `propose.py`) out | Headless Claude or Codex, a fresh session per block; the `/opt-run` skill in a person's session; a scripted advisor in tests |
| Executor | `propose.py` and the data before the block in; proposals out | Sandboxed rerun (benchmark); the person's own machine (real use) |
| Lab | Pending rows in; values out | `Evaluator` over a registered task; a person filling in `data.csv` |

## Settings humans own

All in `problem.toml`; nothing else constrains the agent.

| Setting | Meaning |
| --- | --- |
| `[budget] total` | Hard limit on evaluations |
| `[budget] block_size` | Suggested block size; the agent sizes blocks against the total and remaining budget, and says why |
| `[budget] round_size` | Points the lab measures before results come back; default 1 (see Rounds) |
| `view` | What the advisor is given: `summary`, `raw` or `both` |

## Rounds inside a block (decided 2026-11-03)

The block is the advisor's decision unit; the **round** is the lab's
acquisition unit. Within a block, the executor calls the block's proposer
once per round, on all the data so far, until the block is used, with no
model calls in between. How many points a round holds is set by the
people running the campaign, in `problem.toml`:

- **Expensive, serial experiments and a cheap optimizer**: `round_size = 1`,
  so the optimizer refits after every result.
- **A parallel lab** (a plate, several reactors, a cluster queue): a larger
  `round_size`, so many points go out and come back together. A round as
  large as the block is pure batch.

`round_size` defaults to 1, and every benchmark task uses 1 (the parallel-vial
reaction example uses 8). That restores the
benchmark agent's feedback within a block. It also changes replay, whose
blocks have so far been a single batch, so the 2026-11-03 MOF replays
become the batch-only condition (05).

`propose.py` sets `SIZE` and defines `propose(campaign, q)`, returning the next q points,
instead of writing `proposals.csv` once. It stays a stateless function of
the data, as `agentoptimizer.batch` already is. Methods with internal state
(CMA-ES path, TuRBO length) rebuild it from the block's rounds. Reproduction
replays the rounds with the recorded values and requires the same points at
each one. The last round of a block may be smaller, to fit the block.

## Open decisions

### 1. The prompt

`prompts/optimization_agent.md` and `prompts/advisor.md` carry the same
policy in two forms. Merge them into one policy plus the per-block procedure
(`advisor_headless.md`). This changes the benchmark condition, so the C1–C4
runs are redone under the loop and the old ones are reported as the legacy
agent.

### 2. What happens to `optimizers/`

The stateful ask/tell wrappers exist for the live session. With rounds, the
building blocks are `agentoptimizer.batch`. Sequential local methods
(BOBYQA, Nelder–Mead, L-BFGS-B) need a stateless form: a restart from the
incumbent with a model or simplex rebuilt from recent points. Otherwise they
are dropped. Recommended: port BOBYQA, the one the agent used most, and drop
the rest unless runs show a need.

## What stays, goes and changes

| Piece | Fate |
| --- | --- |
| `blind_runner/evaluator.py` (journal, budget, documents) | Stays; it is the simulated lab |
| `execution.py`, `preflight.py` (sandbox, checks) | Stay; one policy, one probe set |
| `replay.py` (block loop, checks, reproduction, audit) | Becomes the runner; gains rounds and constraints |
| `runner.py` (benchmark main, bridge, harness copy) | Goes; its CLI and audit helpers move into the loop runner |
| `agent_harness/`, `.bridge/`, `write_stdin` keep-alive | Go |
| `optimizers/` | See open decision 2 |
| `advisor/`, `batch/` | Stay; `advisor` gains `view` and constraints |
| `run_agent.py`, `run_replay.py` | One entry point, `scripts/run_loop.py` |
| `/optimize`, `/opt-run` | `/opt-run` for real campaigns; `/optimize` launches and reports loop runs |
| `run_campaign.py`, `run_sweep.py` (non-LLM baselines) | Stay; `run_batch_baselines.py` folds into them |
| `agent_runs.py` | One run type |

Constraints (phase 5 of 05) move up. Six registered tasks are constrained
(`pymoo_g04`, `pymoo_g07`, three `constrained_*`, `opv_pce`), and the suite
needs them.

## Slices

Each slice is tested with a scripted advisor and needs no model calls,
unless marked otherwise.

| Slice | Adds | Tests |
| --- | --- | --- |
| 1 | `[budget] round_size` and `propose(campaign, q)`; the executor loops rounds; reproduction replays them; the policy sizes blocks against the budget | Scripted block of 20 in rounds of 1 matches a hand-run loop; tampered values refused |
| 2 | `view` setting: summary, raw or both | Raw view writes no summary and `summarize` refuses |
| 3 | Constraints through problem, data, summary, `batch` and the lab | A constrained task runs three blocks |
| 4 | One policy prompt; one runner and entry point; `agent_runs.py` and the skills on it | Launch smoke test; benchmark preflight passes on the loop workspace |
| 5 | Remove `runner.py` benchmark path, `agent_harness/`, `optimizers/` (per open decision 2) and their tests | Full test suite; docs and AGENTS.md layout |
| 6 | Pilot: 3 suite tasks and MOF under the loop, 3 seeds (model calls) | Compared with the legacy agent runs |

## Risks

- **Comparability.** Every C1–C4 number gets rerun. The legacy runs stay
  in the release as the earlier condition, not as the main result.
- **Model calls on long budgets.** The agent still decides how long a
  block is, and small blocks on a 2,000-evaluation task mean many calls.
  The policy sizes blocks against the budget; cost per run is reported
  with the scores.
- **Refit cost.** Rounds of 1 refit a GP at every evaluation, as the
  benchmark's GP-BO already did. Fine for the 100-evaluation pools; to
  watch on the 2,000-evaluation suite.
