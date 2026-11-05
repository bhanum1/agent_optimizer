# 06 · The agent loop

Decided 2026-11-05: the loop in `src/agentoptimizer/loop/` (`scripts/run_loop.py`)
is the only agent loop. It replaces the advisor design of 2026-11-03 (`advisor/`,
`batch/`, `blind_runner/`, `prompts/advisor*.md`, `/opt-run`, `/optimize`,
`examples/`), which is in git history before this change. Real-lab campaigns
are not supported until they are rebuilt on this loop.

## Why

The advisor wrote a stateless `propose.py` each block that refit everything from
the data. That suits batch proposers but not optimizers whose value is in their
state (CMA-ES's covariance, a trust region, a local solver's model). The loop
keeps an ask/tell optimizer's state between blocks, so the agent can continue a
method exactly, switch, or run a designed experiment, and decide only how long
each block runs.

## Structure

```text
runner (holds objective, budget, data.csv, decisions.md)
loop:
  1. skill call   a fresh agent reads data.csv / decisions.md / context.md,
                  runs offline analysis, may write work/<opt>.py
                  -> proposal.json, decision entry, context.md
  2. validate     script exists, args parse, 1 <= n_points <= remaining (no early stop)
  3. record       append the decision entry to decisions.md
  4. block        sandboxed optimizer: load state -> n_points x (ask -> runner
                  evaluates -> tell) -> save state; rows appended to data.csv
  5. on crash     runner entry in decisions.md: evaluations used, traceback tail,
                  whether saved state is usable
end: trace.csv / meta.json / decisions.md in results/, audit
```

The objective never enters the sandbox. The agent's only route to a new `y`
is a proposed block.

## Optimizer protocol

An optimizer is a class in one file: `__init__(space, **args)`, `ask(n)`,
`tell(X, y, g)`, `save(path)`, classmethod `load(path)`. A fixed `host.py` runs
it inside the sandbox and speaks JSON lines with the runner, so library and
agent-written optimizers use the same path. `continue` loads the saved state;
`restart` constructs fresh, optionally told every earlier evaluation.

Solvers that call the objective themselves (scipy, Py-BOBYQA) subclass
`lib/_local.Local`: the solver runs in a thread, and `continue` resumes exactly by
replaying the recorded calls. `fixed` holds coordinates at given values so a
solver can work on a subspace.

## The skill

`SKILL.md` frames each block as buying either knowledge that would change the
choice of method or progress from what is known: name the open question,
answer it from the data first, check the response's shape before trusting a
model, run a designed experiment when the data cannot answer, exploit what is
confirmed, and finish the campaign in exploitation. It names no specific
methods and no problem-specific examples.

## Open

- Constraints and noise are supported; real-lab (human-in-the-loop) campaigns are not.
- Figures for loop runs read `results/` (`make_comparison_plots.py --agent-run`).
