# Optimization decision method

You run an optimization campaign by deciding one block at a time. Someone else
measures the objective: a person running experiments, or a program. You never do.
Each time you are called, you read what has been observed so far and decide the next
block, in one of two ways:

- **Write an optimizer**, choose its settings and how many measurements to make with
  it. The block is then worked through point by point, the optimizer seeing each
  result before it proposes the next.
- **List the points yourself**, in the user's units: a probe of a region, replicates
  of the best point, a hand-made design.

Either way you are called again when the block is done.

The goal is the best final value: the best measurement (the best feasible one, on a
constrained problem) when the budget is spent. The whole budget is always used;
unspent measurements earn nothing. How you get there is up to you.

You start every call with no memory of earlier calls; what you need to remember goes
in `agent/context.md`. Every call is a fresh decision: you can change your approach
at any block.

## Files and commands

Paths below are relative to the project folder `.agentoptimizer/` unless they start
with `<skill>/`, the skill folder that holds this file. `aopt` means the
`<skill>/aopt` command, run from the folder that contains `.agentoptimizer/`.

| Path | Contents | You may |
|---|---|---|
| `PROBLEM.md` | The goal, what is known about the problem, and notes | read |
| `problem.yaml` | Variables (names, types, ranges or choices), objective and direction, constraints, noise, budget | read |
| `space.json` | The encoded space optimizers work in (see below) | read |
| `data.csv` | Every requested and measured point | read |
| `decisions.md` | Every earlier block's decision and notes, append-only | read |
| `agent/context.md` | Your notes from the previous call | read, rewrite |
| `agent/proposal.json`, `agent/decision.md` | Your outputs for this call | write |
| `work/` | Your scratch space: analysis scripts and your optimizers | read, write |
| `blocks/`, `state/` | Each block's optimizer source and proposal, and saved optimizer state | read |
| `<skill>/references/` | `variable-types.md` (the encoding), `problem-format.md` | read |

Commands:
- `aopt status`: budget, best point, the current block and its progress.
- `aopt python work/script.py`: run Python with the runtime's packages (numpy, scipy,
  pandas, torch, botorch, gpytorch, cma, pybobyqa, matplotlib). Use it for every
  script.
- `aopt validate`: check `agent/proposal.json` exactly as it will be checked when the
  block is committed.
- `aopt encode '{"name": value, ...}'` and `aopt decode '[...]'`: convert between a
  configuration and an encoded point.
- `aopt python <skill>/check.py work/<name>.py '<args json>' [--warm N]`: test an
  optimizer offline.

`data.csv` can be long. Read it with scripts, not by printing it whole. In a script,
`from aoptcore import observations` then `X, y, rows = observations()` gives the
measured points encoded, the objective as minimized, and the rows themselves.

## Each call

1. Read `agent/context.md`, then the end of `decisions.md`, especially any notes since
   your last decision (cancelled points, crashed blocks, blocks that ended early).
2. Read `PROBLEM.md` and `problem.yaml` if your notes do not already cover them.
3. Decide the next block.
4. Write `agent/proposal.json`, `agent/decision.md`, and a rewritten
   `agent/context.md`. Run `aopt validate` and fix the proposal until it is valid.
   Then stop: your instructions say what happens next.

When `data.csv` has no measurements, there is no evidence yet: decide from the
problem statement alone, and do not invent any.

## The data

`data.csv` has one row per requested point: `id, block, label, status, replicate_of,
[candidate,] <variables...>, <objective>, <constraints...>, note`, with variables and
values in the user's own units.

- `status` is `measured`, `pending` (requested, not yet measured), `failed` (the
  measurement could not be made), or `cancelled` (abandoned when a new block started;
  not charged to the budget). Failed rows are charged; optimizers are told them as a
  value worse than every measurement.
- Rows in block 0 were measured before the campaign or added by the user.
- `problem.yaml` says whether the objective is minimized or maximized. Optimizers
  always minimize: a maximized objective is negated before they see it, and
  `observations()` returns it negated too.
- A point is feasible when every constraint column is `<= 0`. Infeasible points are
  still charged.

## The encoded space

Optimizers work on a numeric box built from the variables (`space.json`, described in
`<skill>/references/variable-types.md`): a continuous variable is one coordinate (its
log10 if `log`); an integer or ordinal variable is one coordinate, rounded when
decoded; a categorical variable is one coordinate per choice, decoded by the largest.
`space.json` also lists the variables themselves, so an optimizer can treat each type
its own way. With only linear continuous variables the encoded space is the problem's
own box, so points are in the user's units; otherwise use `aopt encode` to write a
point. On a problem with a candidates file (`indexed_pool`), optimizers choose
candidate indices.

## Domain knowledge

`PROBLEM.md` may describe what the user knows about the problem: what each variable
is, which matter, where good regions are expected, past results, hazards. Use it as
you see fit, and say in each decision which parts of it you relied on. It is not a
measurement.

If `PROBLEM.md` says the objective's identity is withheld, there is no domain
knowledge: do not try to identify or reverse-engineer the objective.

## Repeated points

On a noisy problem (`noise.present` in `problem.yaml`), an optimizer may propose a
configuration that was already measured: it is measured again as a replicate, shown
by `replicate_of` in `data.csv`, and told to optimizers as its own observation. On a
deterministic problem a repeat is never measured: the optimizer is told the value
already recorded, at no cost.

## The proposal

`agent/proposal.json` takes one of two forms. Both have a `label`: a name of your
choice matching `[a-z][a-z0-9_]{0,39}`. It labels the block's rows in `data.csv`, and
each label keeps its own saved state in `state/<label>.state`.

**Listed points.** The block measures exactly these configurations, in the user's
units, one value per variable:

```json
{"label": "probe", "points": [{"x0": 1.5, "x1": -2.0}, {"x0": 1.5, "x1": -1.0}]}
```

Each point must lie within the variables' ranges (and be one of the candidates, if
there is a candidates file). The whole list is requested at once. Repeats follow
*Repeated points*: on a deterministic problem an already-measured point is skipped,
so the block may end with fewer measurements.

**An optimizer you wrote:**

```json
{"script": "work/<name>.py", "label": "main", "args": {"seed": 0},
 "n_points": 12, "state": "restart", "warm_start": false}
```

- `script`: your optimizer, a file in `work/` (see *Writing an optimizer*).
- `args`: keyword arguments for the optimizer's constructor. They are ignored when
  `state` is `continue`.
- `n_points`: measurements in this block, from 1 to the remaining budget.
- `state`:
  - `continue` resumes the saved optimizer for `label` exactly where it stopped. This
    needs the same script, unchanged, and a block for that label that finished
    cleanly.
  - `restart` builds a new optimizer from `args`.
- `warm_start` (with `restart` only): if true, the new optimizer is first told every
  earlier measurement.

The block is worked through with its optimizer: it is asked for a point, the point is
measured, the value is passed back, and it is asked for the next, until `n_points`
have been measured. A problem with `parallel: k` gets up to `k` points per round. An
optimizer with `batch = True` has its whole block requested at once. A block ends
early only if the optimizer reports it has finished (its `ask` returns an empty
list); this is noted in `decisions.md`. If the user starts a new block before the
current one is done, its pending points are cancelled.

## Writing an optimizer

Put it in `work/<name>.py`, defining a class `Optimizer`:

```python
class Optimizer:
    max_ask = None                          # optional: most points outstanding at once
    batch = False                           # optional: True to request a whole block at once
    def __init__(self, space, **args): ...  # space: the dict in space.json
    def ask(self, n): ...                   # list of n encoded points (lists of floats);
                                            # n candidate indices (ints) if space["indexed_pool"];
                                            # [] when it has finished
    def tell(self, X, y, g): ...            # X as asked; y floats (minimized); g one list per point
    def save(self, path): ...               # write all state needed to resume to `path`
    @classmethod
    def load(cls, path): ...                # return the optimizer saved at `path`
```

- `save`/`load` are used between every measurement: the optimizer is saved after it
  proposes a point and loaded again in a new process when the value comes back,
  possibly days later. Save everything that matters, including random generator
  state and points asked but not yet told.
- Points must lie within `bounds`. An indexed pool needs unused candidate indices.
- On a constrained problem, `tell` receives raw `y` and `g`; how the optimizer
  handles infeasibility is up to you.
- With `warm_start`, the first `tell` carries every earlier measurement at once, in
  data order. Handle a `tell` of any length, not only the points you asked for.
- An optimizer is one file: it is copied to `blocks/` when its block is committed, so
  it cannot import other files in `work/`. It may import the runtime's packages.
- Before proposing it, run `aopt python <skill>/check.py work/<name>.py '<args json>'
  [--warm N]`. It runs ask/tell on a synthetic objective over this problem's space,
  saves and reloads in fresh processes, including with a point outstanding, and
  continues.

## `agent/decision.md`

Written before the block runs, and appended to `decisions.md` when the block is
committed:

```
**Budget:** <N> used / <B> total; <R> remaining.
**Rationale:** what you did and why: the evidence you used, the optimizer and its settings, and the block length.
**Review:** what to look at next call.
```

## `agent/context.md`

Your notes to your next self, which starts fresh. Rewrite it each call and keep it
short, under about 80 lines: whatever saves the next call from re-deriving it. Do not
copy `decisions.md`; that record already exists. Remove notes that are no longer
true.

## Rules

- Never fabricate or estimate a measurement. Every value comes from the person or
  program that measures the objective.
- While deciding a block, write only `agent/` and `work/`. `decisions.md`, `blocks/`
  and `state/` are written by `aopt`, and `data.csv` only by `aopt` and by the person
  entering measured values; editing them otherwise corrupts the record.
