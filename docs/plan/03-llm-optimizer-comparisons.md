# 03 · LLM-optimizer comparisons

Supports claim C3. Our agent decides once per block and runs library
optimizers in between, so its LLM calls scale with decisions, not with
evaluations. The comparisons test whether that costs quality, and what it
saves.

## Direction A: their methods on our tasks

| Method | LLM role | Source | Code |
| --- | --- | --- | --- |
| Direct LLM proposer | Proposes each point from the history in context (OPRO / LLAMBO style) | Yang et al. 2024; Liu et al. 2024 | Simple to reimplement |
| Per-evaluation agent | Coding agent with a workspace, submits one candidate per round | AgenticBBO-Bench, arXiv 2610.12183 | To verify |
| Centaur | LLM sees CMA-ES state and proposes on a share of trials | Ferreira et al., arXiv 2603.24647 | Appears public (`ferreirafabio/autoresearch-automl`); to verify |
| Sara + lenz | Agent drives a BoTorch backend, changes acquisition mid-run | Brunzema et al., arXiv 2608.00316 | To verify |
| One-shot LLM selection | LLM picks one optimizer from the task description and budget, then runs it | Modeled on BBOWP-Bench, arXiv 2608.02612 | Reimplement as an ablation of our agent |

## Direction B: our agent on their benchmarks

- AgenticBBO-Bench BBOB setting: 24 functions, 20 initial + 100 new
  evaluations. Shows our agent outside our own task suite and at a small
  budget.
- Others only if their harness is public and the budget model maps cleanly
  onto ours.

## Cost accounting (all LLM methods)

Report per run: LLM calls, input and output tokens, wall-clock time, and
dollar cost where available, next to the score. Fix the backbone model per
comparison, or report each method with the same set of models.

## Fairness rules

- Same budgets, seeds and blindness. A method that needs a task description
  gets the same description for every method that can use one; report
  "blind" and "described" settings separately.
- Count only objective calls against the budget; report LLM cost separately.

## To verify

- Code and licenses for AgenticBBO-Bench, Centaur and Sara.
- Whether AgenticBBO-Bench's scoring (70% normalized best-so-far area, 30%
  final best) can be computed from our traces.
