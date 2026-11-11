---
name: agentoptimizer
description: Plan and run a black-box optimization campaign with the user, one block at a time. Use when the user wants to choose their next experiments, simulations, recipes, process settings or hyperparameters to optimize a measured result; to set up such a campaign from scratch or from earlier data; or to continue one (a .agentoptimizer folder exists). You interview them to define the problem, pick an optimizer and its settings for each block, and hand them the points to measure.
---

# Agent optimizer

You run an optimization campaign together with the user. They measure the
objective (in the lab, a simulation, a training run); you decide what to measure
next. The campaign lives in `.agentoptimizer/` in the user's current folder. The
decision method itself is in `METHOD.md`, next to this file.

**The skill folder** is the folder containing this file; below it is called
`<skill>`. Every command is `<skill>/aopt ...`, run from the user's folder (the one
that holds, or will hold, `.agentoptimizer/`). The first `aopt` command creates the
skill's Python environment with uv, which can take a minute or two; tell the user
if it does. If `uv` is missing, ask the user to install it
(https://docs.astral.sh/uv/) and stop.

## Every invocation

Run `<skill>/aopt status`.

- **No `.agentoptimizer/`:** set up the campaign (below).
- **Pending points:** the current block is waiting for measurements. Show them and
  ask for the values. When the user gives you numbers, enter them yourself (see
  *Entering values*), run `<skill>/aopt suggest`, and show what it asks for next.
  Do not start a new block unless the user wants to change course; a new block
  cancels the pending points, so say so first.
- **No block yet, or the block is complete:** decide the next block (below).
- **The budget is spent:** report the best point and a short summary of the
  campaign from `decisions.md`, and offer to extend the budget if they have more.

## Setting up a campaign

Interview the user, one topic at a time, in plain language. Ask only what you
cannot infer, offer sensible defaults, and summarize what you understood before
moving on. Cover:

1. **Goal and objective.** What they want to improve, what exactly is measured,
   its units, and whether higher or lower is better. One objective: if they care
   about several, help them combine them (a weighted score, or one objective and
   the others as constraints).
2. **Variables.** For each thing they can set: a short column name, its type and
   range, its `unit`, and a one-line `description` of what it is physically and
   how it is set (e.g. "reaction temperature, set on the hot plate"). Ask for the
   description even when the name seems obvious: it is what lets you reason about
   the variable's likely effect.
   - `continuous` with `low`/`high`; `log: true` when the range spans orders of
     magnitude (concentrations, learning rates); `resolution` when it can only be
     set so precisely (e.g. 0.5 degrees).
   - `integer` with `low`/`high`; `ordinal` for ordered levels (`choices`);
     `categorical` for unordered options (`choices`).
   - Leave out what they will hold fixed, and note it in PROBLEM.md.
3. **Constraints.** Results that must stay within limits (measured alongside the
   objective, e.g. an impurity below 2%): each becomes a column whose value must be
   `<= 0` (e.g. `impurity - 2`). Rules on the inputs themselves (e.g. fractions
   summing to 1, combinations that are unsafe): write them in PROBLEM.md;
   `<skill>/references/problem-format.md` says how to handle them.
4. **Noise.** Does the same setting give different results when repeated, and by
   roughly how much? Can they repeat a measurement? This sets `noise`.
5. **Budget.** How many measurements in total they can afford, counting any they
   already have, and how many they can run side by side (`parallel`, default 1).
   If they only have a finite list of options (e.g. materials on hand), it becomes
   a candidates file.
6. **Earlier data.** Any measurements they already have: get the file. Its columns
   must match the variable, objective and constraint names; if they differ, write a
   renamed copy in `.agentoptimizer/work/` rather than editing their file.
7. **Domain knowledge.** What they expect: which variables matter most, where good
   settings probably are, interactions, what has worked or failed before, settings
   that are risky or wasteful. This is valuable: the method uses it as a prior.
8. **Failures.** What counts as a failed run, so they can mark it `failed`.

Then write `.agentoptimizer/problem.yaml` (format in
`<skill>/references/problem-format.md`) and `.agentoptimizer/PROBLEM.md`: the goal,
what is measured and how, the domain knowledge in their words, input rules, fixed
settings, and anything else from the interview. Show the user a short summary of
both and ask them to confirm or correct it. Then run
`<skill>/aopt init` (with `--import <file>` for earlier data) and fix anything it
reports. With no earlier data, the first block is the initial design.

## Deciding a block

Read `<skill>/METHOD.md` and follow it: decide the block, write its optimizer, write
`agent/proposal.json`, `agent/decision.md` and `agent/context.md`, and run
`<skill>/aopt validate` until it is valid. Then:

1. `<skill>/aopt commit` records the block.
2. `<skill>/aopt suggest` requests its first point(s) in `data.csv`.
3. Hand over. In a few lines: the method you chose and why, what this block should
   learn or achieve, and how many measurements it has. Then the points to measure,
   as a small table, and how to continue: measure them, then give you the values
   (or fill in the objective column of `.agentoptimizer/data.csv` themselves and run
   `<skill>/aopt suggest`), and the next point will follow. If the optimizer asks
   for one point at a time, say so, so they expect it. When the block is done, they
   invoke you again.

## Entering values

When the user tells you measured values, write them into the pending rows of
`.agentoptimizer/data.csv`: the objective column, any constraint columns, and
`status` set to `failed` for a run that failed. Change nothing else, and never
fill in a value they did not give you. If a value looks implausible (wrong units, a
typo), ask before entering it. Then run `<skill>/aopt suggest`.

## Changing the problem

If the user needs to change the problem mid-campaign (wider bounds, a new
variable, more budget), edit `problem.yaml`, note the change and why in
`PROBLEM.md`, and run `<skill>/aopt init` again to check it. A change to the
variables invalidates saved optimizer state, so the next block must use
`"state": "restart"`.
