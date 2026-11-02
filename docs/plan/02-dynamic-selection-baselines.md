# 02 · Dynamic-selection baselines

Supports claim C2: the agent's gain comes from adaptive selection, not only
from choosing one good method. Every method here uses the same optimizer
portfolio the agent can reach, unless noted.

## Methods (proposed)

| Method | What it does | Needs training data | Notes |
| --- | --- | --- | --- |
| Nevergrad NGOpt | Hand-designed selector that picks and chains optimizers from dimension, budget and noise | No | The baseline reviewers will expect; uses Nevergrad's own portfolio, so report it as such |
| Feature-based selection with probing | Spend a probing budget on landscape features (pflacco) or short runs of each candidate, then give the remainder to one optimizer | Yes (selector model) | ABBO-style; probing evaluations count against the budget |
| Single-switch dynamic algorithm selection | Start with optimizer A, switch to B at a learned point | Yes | Vermetten et al. (arXiv 2006.06586) |
| Online portfolio / bandit | Run several optimizers and allocate evaluations by observed progress (e.g. UCB) | No | Assumption-light comparator |
| Rule-based switching | Switch on stagnation (as in the Polymorphic Metaheuristic Framework, arXiv 2505.13808); global-then-local hybrids | No | Hand rules; document thresholds |
| CMA-ES with restarts (IPOP/BIPOP) | Restart strategy inside one family | No | Shows how much a strong single-family method recovers |
| SBS and VBS | Best single optimizer overall; per-task oracle | — | Already computed |
| Best fixed single switch (oracle) | Best A→B pair and switch point per task, chosen in hindsight | — | Upper bound for one switch |

## Fairness rules

- Same task suite, budgets and seeds as the agent runs.
- Every objective call counts, including probes, feature computation and
  restarts.
- Methods that need training data are trained leave-one-task-out (or
  leave-one-family-out), never on the task they are scored on.
- Hyperparameters: library defaults unless a method's paper prescribes
  otherwise; record versions.

## Interface (to settle in the first implementation PR)

A selector wraps other optimizers, so the plain ask/tell interface is not
enough. Questions the first PR must answer:

- How a selector hands state to the next optimizer (best point, population,
  covariance) and which optimizers can accept a warm start. Advisor mode
  needs the same warm start; see the table in
  [05-advisor-mode.md](05-advisor-mode.md) and share the adapters.
- Whether selectors run through `run_campaign.py` unchanged (if they expose
  ask/tell outward) or need their own runner.
- How the trace records which inner optimizer produced each evaluation, so the
  same switching plots work for the agent and for selectors.

## To verify

- NGOpt variant and Nevergrad version to use (the evaluator environment
  already pins `nevergrad==1.0.12` for noisy tasks).
- Whether ABBO code is public (arXiv 2509.25592).
- Training-data source for learned selectors: the existing baseline sweeps
  (old repo, `results_linux/`) are a candidate.
