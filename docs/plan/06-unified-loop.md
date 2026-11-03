# 06 · One loop for benchmarks and real campaigns

The repository had two systems built on the same policy:

- **Benchmark mode** (`run_agent.py`, `/optimize`): one live session drove
  `AgentTaskHarness`. It evaluated points one at a time through a file
  bridge and kept optimizer processes alive with `write_stdin`.
- **Advisor mode** (`opt_run.py`, `/opt-run`, replay): each block, the agent
  reads a campaign folder and writes `propose.py`. That script fits the data
  and proposes the block, which a lab (or the replay runner) evaluates.

The second is now the design everywhere (decided and built 2026-11-03). The
agent receives data, as a summary or raw. It writes the optimizer for the
next block from building blocks, or from scratch. The optimizer fits the
data and proposes points, round by round. Acquisition measures them, and the
loop repeats. A benchmark is this loop with a simulated lab. Benchmark mode
is retired; its published runs are the legacy condition.

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
| `[budget] total` | Hard limit on evaluations; required |
| `[budget] block_size` | Suggested block size, default 10% of the total; the agent sizes blocks against the total and remaining budget, and says why |
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

## Decisions made (2026-11-03)

- **One prompt.** `prompts/optimization_agent.md` is retired (it stays at
  the `pre-manuscript-snapshot` tag of the source repository).
  `prompts/advisor.md` is the policy and `advisor_headless.md` the
  per-block procedure. The C1–C4 runs are redone under the loop, and the
  old ones are reported as the legacy agent.
- **`optimizers/` stays, for the baselines only.** `run_campaign.py` and
  `run_sweep.py` run its ask/tell wrappers; the agent no longer gets a copy.
  Its building blocks are `agentoptimizer.batch`. Sequential local methods
  (BOBYQA, Nelder–Mead) have no batch form; with rounds of 1 the agent can
  write one in `propose()` (for example a BOBYQA step from the incumbent),
  so none is ported until runs show the need.
- **Defaults.** `run_agent.py` defaults to Claude Opus 5.5 at medium effort,
  the C6 setting; `--agent codex` uses GPT-6 Astra.
- **Methods are designed from hypotheses about the landscape.** A first
  live run (sphere10, Opus at max effort, `runs/run-de54a5ff0684`, stopped
  after 2 blocks at $2.80) built its own design and compared methods in
  simulation, but checked only how well a GP fit, never what the data said
  about structure: a bowl with a free centre fits its first 12 points
  exactly, and it never tried one. The policy now asks, each block, for
  hypotheses about the landscape that bear on the choice of optimizer,
  tested on the data, and for a method designed for them. The aim is still
  the best optimizer, not identifying the function, and the policy names
  no model families, so it hints nothing about the tasks. The GP
  diagnostic in `summary.json` is removed: it chose the agent's first view
  of the data for it. Structure-seeking may favour the synthetic tasks, so
  synthetic and real tasks are reported separately.

## Status (2026-11-03)

Built: rounds (`[budget] round_size`, `SIZE` and `propose(campaign, q)`),
the runner proposing later rounds in the sandbox, and one runner
(`blind_runner/runner.py` with `loop.py`, `agent_cli.py`, `campaign.py`).
Removed: `agent_harness/`, the bridge, the benchmark preflight and prompt,
`run_replay.py`, the evaluator's workspace artifacts and its "fixed"
document rule. A scripted advisor (no model) ran sphere10 and the MOF pool
through the real sandbox, with every later round proposed there and a clean
audit.

| Piece | Now |
| --- | --- |
| `blind_runner/evaluator.py` | The simulated lab: journal, budget, `decisions.md` append-only |
| `execution.py`, `preflight.py` | One sandbox policy (writable `campaign/`, `work/`, `.tmp/`), one probe |
| `loop.py` | Blocks and rounds, checks, reproduction (was `replay.py`) |
| `agent_cli.py`, `campaign.py`, `runner.py` | Headless advisor; the anonymous campaign; setup, audit, `main` |
| `scripts/run_agent.py`, `agent_runs.py`, `/optimize` | One entry point and one kind of run, `runs/run-<id>/` |

Still to do:

- **Constraints** through problem, data, summary, `batch` and the lab. Six
  registered tasks are constrained (`pymoo_g04`, `pymoo_g07`, three
  `constrained_*`, `opv_pce`) and the suite needs them; the runner refuses
  them until then.
- **`view`**: summary, raw or both, as a condition.
- **Figures** for loop runs: `make_comparison_plots.py` reads only legacy
  `results/agent_run/` folders.
- **`batch.space_filling` ignores its seed on pools** (05); fix before the
  batch baselines are final.
- **Pilot**: three suite tasks and MOF under the loop, three seeds, against
  the legacy runs (model calls; ask first).

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
