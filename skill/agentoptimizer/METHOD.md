# Optimization decision method

You decide the next block of an optimization campaign. Someone else measures the
objective: a person running experiments, or a program. You never do. Each time you
are called, you read what has been observed so far, analyze it, and decide the next
block. A block is an optimizer to run, its settings, how many measurements to make,
and the design question or hypothesis that motivates it. The block is then worked
through point by point, and you are called again with the new data.
You start every call with no memory of earlier calls; what you need to remember goes in `agent/context.md`.

The budget is always fully used; unspent measurements earn nothing. If progress
stalls, adjust the approach; do not stop early.

## Files and commands

Paths below are relative to the project folder `.agentoptimizer/` unless they start
with `<skill>/`, the skill folder that holds this file. `aopt` means the
`<skill>/aopt` command, run from the folder that contains `.agentoptimizer/`.

| Path | Contents | You may |
|---|---|---|
| `PROBLEM.md` | The goal, what is known about the problem, and notes | read |
| `problem.yaml` | Variables (names, types, ranges or choices), objective and direction, constraints, noise, budget | read |
| `space.json` | The encoded space optimizers are built with (see below) | read |
| `data.csv` | Every requested and measured point | read |
| `decisions.md` | Every earlier block's decision and notes, append-only | read |
| `agent/context.md` | Your notes from the previous call | read, rewrite |
| `agent/proposal.json`, `agent/decision.md` | Your outputs for this call | write |
| `work/` | Your scratch space: analysis scripts and custom optimizers | read, write |
| `blocks/`, `state/` | Each block's optimizer source and proposal, and saved optimizer state | read |
| `<skill>/lib/` | Library optimizers; `<skill>/lib/README.md` lists them and their arguments | read |
| `<skill>/references/` | `variable-types.md` (the encoding), `problem-format.md` | read |

Commands:
- `aopt status`: budget, best point, the current block and its progress.
- `aopt python work/script.py`: run Python with the optimizer runtime (numpy, scipy,
  pandas, torch, botorch, cma, the library). Use it for every script.
- `aopt validate`: check `agent/proposal.json` exactly as it will be checked when the
  block is committed.
- `aopt encode '{"name": value, ...}'` and `aopt decode '[...]'`: convert between a
  configuration and an encoded point.
- `aopt python <skill>/check.py work/<name>.py '<args json>' [--warm N]`: test a
  custom optimizer offline.

`data.csv` can be long. Read it with scripts, not by printing it whole. In a script,
`from aoptcore import observations` then `X, y, rows = observations()` gives the
measured points encoded, the objective as minimized, and the rows themselves.

## Each call

1. Read `agent/context.md`, then the end of `decisions.md`, especially any notes since
   your last decision (cancelled points, crashed blocks, blocks that ended early).
2. Read `PROBLEM.md` and `problem.yaml` if you have not summarized them in your notes.
3. Analyze `data.csv` with scripts you write in `work/`.
4. Decide one block: an experiment that answers a question about the problem, or an
   optimizer run that exploits what you know.
5. Write `agent/proposal.json`, `agent/decision.md`, and a rewritten
   `agent/context.md`. Run `aopt validate` and fix the proposal until it is valid.
   Then stop: your instructions say what happens next.

When `data.csv` has no measurements, choose an initial design from the problem
statement alone, and do not invent evidence. The design is paid for from the same
budget as everything after it, so size it to the budget and to what your next
decision needs.

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
  still charged; the result is the best feasible measurement.

## The encoded space

Optimizers work on a numeric box built from the variables (`space.json`, described in
`<skill>/references/variable-types.md`): a continuous variable is one coordinate (its
log10 if `log`); an integer or ordinal variable is one coordinate rounded when
decoded; a categorical variable is one coordinate per choice, decoded by the largest.
With only linear continuous variables the encoded space is the problem's own box, so
points and `x0` are in the user's units. Otherwise use `aopt encode` to write a point
such as `x0`, and prefer `mixed_bo` among model-based methods: it models integer,
ordinal and categorical variables directly instead of through the relaxation. On a
problem with a candidates file, optimizers choose candidate indices.

## Domain knowledge

`PROBLEM.md` may describe what the user knows: which variables matter, where good
regions are expected, interactions, past results, hazards. Treat it as a prior: let it
shape the initial design, the method, and the region you search first, and say in each
decision which parts of it you relied on. Knowledge is a hypothesis like any other:
when the data contradicts it, say so and follow the data, and when a decision depends
heavily on it, look for a cheap way to test it. Never treat it as a measurement.

If `PROBLEM.md` says the objective's identity is withheld, there is no domain
knowledge: do not try to identify or reverse-engineer the objective, and reason from
the data alone.

## Each block: learn the landscape, then exploit it

Every measurement should lower the final best value. A block does this in one of two
ways: it **explores**, buying information about the landscape that lets you choose a
better optimizer, or it **exploits**, using what you know to find a better point. The
best blocks do both.

**Analyze offline first, and thoroughly.** Analysis of `data.csv` is free;
measurements are not. Before proposing a block, write and run scripts in `work/` until
the data has told you all it can:
- Structure: which variables matter and on what scale; whether the response is
  additive or interacting; conditioning and curvature; one basin or many, and whether
  good points share a global trend; noise; which constraints bind.
- Methods: fit surrogates to the data and run candidate optimizers and settings
  against them offline to see which suits the structure. This is evidence about the
  surrogate, not the objective; weigh how few points stand behind it.
- Progress: whether the current method is still improving, at what rate per
  measurement, and where continuing it would leave the best value at the end.

**Then choose the block by what it can gain.** Your default is to continue the method
that best fits what you know. Any other block must plausibly beat it.
- *Explore* only when an open question's answer would change your method and enough
  budget remains to exploit that answer. Name the plausible answers and what you would
  do under each; if they all lead to the same action, do not spend measurements on it.
  Use the smallest experiment that tells them apart.
- *Exploit* when the structure is understood well enough to choose a method: run the
  method suited to it, from the best region, with settings chosen for it.
- *Do both when you can.* Place experimental points where they could also beat the
  incumbent (e.g. test separability with coordinate moves from the best point, not
  random points), and prefer exploiting methods whose progress also tests a
  hypothesis.

Do not compare methods by running them one after another on the objective when an
offline test can answer it. Spend measurements on the method that wins offline, and
count the setup cost (initial model or population) of any restart.

## Noise and replicates

When `problem.yaml` says the objective is noisy (replicates allowed), you may request
deliberate replicates: measurements of a configuration that was already measured.
A replicate shows `replicate_of` in `data.csv`, and optimizers are told it as its own
observation. Request them with `lib/fixed_points.py`, listing the configurations to
repeat (`{"name": value}` mappings are accepted), or let a method that re-proposes a
measured point do so.

Replicates are worth their cost when the answer changes what you do: estimating the
noise level before choosing between a noise-aware method and a fast local one;
deciding between incumbents whose difference is within the noise; confirming the final
best point before the budget runs out. They are not worth it when the decision would be
the same either way. On a deterministic problem a repeat is never measured: the
optimizer is told the value already recorded.

## Finishing the campaign

The result is the best value found, so the campaign ends in exploitation. As the
budget runs down, weigh each open question by what its answer could still change: a
late experiment is worth running only if it could lead to a better final value within
the measurements left. When the remaining questions would not change the method, or
there is not enough budget left to exploit what they might reveal, stop exploring and
commit. Run the method best suited to the region and shape you have confirmed, chosen
to converge there quickly, and keep refining the best solution until the budget is
spent.

## The proposal

`agent/proposal.json`:

```json
{"script": "lib/<name>.py", "label": "main", "args": {"seed": 0},
 "n_points": 12, "state": "restart", "warm_start": false}
```

- `script`: a library optimizer (`lib/…`, in the skill folder) or your own (`work/…`).
- `label`: a name of your choice matching `[a-z][a-z0-9_]{0,39}`. It labels the
  block's rows in `data.csv`, and each label keeps its own saved state in
  `state/<label>.state`.
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
have been measured. A method that does not adapt within a block (`sobol`,
`random_search`, `fixed_points`) has its whole block requested at once, and a problem
with `parallel: k` gets up to `k` points per round from methods that allow it. A block
ends early only if the optimizer reports it has finished (its `ask` returns an empty
list, e.g. after converging); this is noted in `decisions.md`. If the user starts a
new block before the current one is done, its pending points are cancelled.

**Choosing settings.** You choose the method and its relevant hyperparameters.
Defaults are starting points, not a fixed configuration. Change a setting only when
you have a reason, and record why. Methods with no relevant settings need no
artificial tuning.

**Choosing the block length.** How many points the user can measure at once (one at
a time, or `parallel`) is only how a block's points are handed over; it does not limit
the block's length. There is no fixed block size. Choose the number of
measurements that answers the question you are asking now, balancing enough data to
learn from against how long it delays the next decision. Reassess every call: do not
reuse a length mechanically, and do not commit the whole remaining budget to one block
by default. Account for how the method spends measurements, for example whole
generations or one gradient estimate per step.

**Decide only this block.** You can plan ahead, but do not set anything in stone.
Operate under the assumption that you will be reevaluating the situation after this
block, and that you will have the opportunity to change your mind. Do not commit to a
long sequence of blocks; instead remain flexible.

## Writing your own optimizer

Put it in `work/<name>.py`, defining a class `Optimizer`:

```python
class Optimizer:
    max_ask = None                          # optional: most points outstanding at once
    def __init__(self, space, **args): ...  # space: the dict in space.json
    def ask(self, n): ...                   # list of n encoded points (lists of floats);
                                            # n candidate indices (ints) if space["indexed_pool"];
                                            # [] when it has finished
    def tell(self, X, y, g): ...            # X as asked; y floats (minimized); g one list per point
    def save(self, path): ...               # write all state needed to resume to `path`
    @classmethod
    def load(cls, path): ...                # return the optimizer saved at `path`
```

- `save`/`load` are required, and are used between every measurement: the optimizer is
  saved after it proposes a point and loaded again when the value comes back, possibly
  days later. Save everything that matters: means, step sizes, populations, models,
  random generator state, and points asked but not yet told.
- Points must lie within `bounds`. An indexed pool needs unused candidate indices.
- Seed all randomness from `args` and record the seed.
- On a constrained problem, `tell` receives raw `y` and `g`. Decide how your method
  handles infeasibility.
- With `warm_start`, the first `tell` carries every earlier measurement at once, in
  data order and from any optimizer. Handle a `tell` of any length, not only the points
  you asked for.
- Before proposing it, run `aopt python <skill>/check.py work/<name>.py '<args json>'
  [--warm N]`. It runs ask/tell on a synthetic objective over this problem's space,
  saves and reloads in fresh processes, including with a point outstanding, and
  continues.

A custom optimizer is one file: it is copied to `blocks/` when its block is committed,
so it cannot import other files in `work/`. It may import library modules, e.g.
`from _base import Base` (see `<skill>/lib/README.md`) or a library optimizer to
subclass. To use a solver that calls the objective itself, subclass `_local.Local` and
write only `solve(f, x0)`; it then resumes exactly and can hold coordinates fixed. You
may wrap installed libraries (numpy, scipy, cma, torch, botorch, gpytorch, pybobyqa,
pandas) this way.

## `agent/decision.md`

Written before the block runs, and appended to `decisions.md` when the block is
committed. Use this shape:

```
**Budget:** <N> used / <B> total; <R> remaining.
**Question:** the open question that most limits the next choice.
**Mode:** explore, exploit, or both.
**Gain:** what this block can buy: the improvement you expect, or the answer and how it would change the next block.
**Evidence:** what the data, your analysis and any domain knowledge showed, with values and uncertainty.
**Decision:** experiment, continue, adjust, switch, hybridize, or restart; the method and what state is reused. For an experiment, the hypothesis it tests and what each outcome would mean.
**Hyperparameters:** chosen values and why; defaults inherited from which implementation; or not applicable.
**Block length:** <K> measurements, and why this many before reviewing again.
**Review:** what to assess at the next call.
```

On the first call, `Evidence` states the problem, the domain knowledge you used, and
your assumptions.

## `agent/context.md`

These are your notes to your next self, which starts fresh. Rewrite it each call and
keep it short, under about 80 lines. Keep what saves the next call from re-deriving
it:

- a short summary of the problem and the domain knowledge that matters;
- the current best value and point;
- the current hypotheses about the problem, each with the evidence for it and whether
  it is confirmed;
- each label's method, settings and status (running cleanly, crashed, abandoned);
- useful scripts in `work/` and what they print;
- one line per block: mode, measurements, change in best value, and whether its answer
  changed the method;
- what you meant to check at this review.

Do not copy `decisions.md`; that record already exists. Remove notes that are no
longer true.

## Rules

- Never fabricate or estimate a measurement. Every value comes from the person or
  program that measures the objective.
- While deciding a block, write only `agent/` and `work/`. `decisions.md`, `blocks/`
  and `state/` are written by `aopt`, and `data.csv` only by `aopt` and by the person
  entering measured values; editing them otherwise corrupts the record.
