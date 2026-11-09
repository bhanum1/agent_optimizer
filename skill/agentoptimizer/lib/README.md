# Library optimizers

Use one with `"script": "lib/<name>.py"`. Optimizers see the encoded space in
`space.json` (see the skill's references/variable-types.md): with only
continuous variables it is the problem's own box, so points and `x0` are in the
user's units; integer, ordinal and categorical variables are relaxed to
continuous coordinates and rounded when decoded. `mixed_bo` models them
directly and is the default model-based choice when the space is mixed. Every optimizer takes `seed`
(default 0) plus the arguments listed in its file's docstring; read it
for exact defaults. Optimizers work in the unit cube internally: step
sizes, radii and lengths are fractions of each variable's range, while
`x0` and points are in real coordinates.

| Name | Method | Spaces | `continue` resumes |
|---|---|---|---|
| `random_search` | Uniform random points, or uniform unused candidates | box, pool | exactly |
| `sobol` | Scrambled Sobol sequence | box | exactly |
| `fixed_points` | The points you list in `points` (encoded lists, `{name: value}` configurations, or candidate indices), then finishes; repeats of measured points are replicates on a noisy problem | box, pool | exactly |
| `cma_es` | CMA-ES (pycma): `x0`, `sigma0`, `popsize`, `options` | box, pool* | exactly |
| `de` | Differential evolution, steady state: `pop_size`, `F`, `CR`, `strategy` | box, pool* | exactly |
| `pso` | Particle swarm: `swarm`, `w`, `c1`, `c2`, `vmax` | box, pool* | exactly |
| `nelder_mead` | Nelder-Mead simplex: `x0`, `step`, `adaptive`, `xtol`, `ftol` | box, pool* | exactly |
| `mixed_bo` | GP Bayesian optimization for mixed spaces (botorch MixedSingleTaskGP): `n_init` | box | from the data (model refitted each step) |
| `gp_bo` | GP Bayesian optimization (botorch): `n_init`, `acquisition` (logei, ucb), `beta` | box, pool | from the data (model refitted each step) |
| `turbo` | TuRBO-1 trust-region BO: `n_init`, `length_*`, `success_tol`, `failure_tol` | box, pool | exactly |
| `lbfgsb` | L-BFGS-B with finite differences (scipy): `x0`, `fixed`, `eps`, `ftol`, `gtol` | box | exactly (replay) |
| `powell` | Powell's method (scipy): `x0`, `fixed`, `xtol`, `ftol` | box | exactly (replay) |
| `bobyqa` | BOBYQA quadratic-model trust region (Py-BOBYQA): `x0`, `fixed`, `rhobeg`, `rhoend`, `npt` | box | exactly (replay) |
| `cobyla` | COBYLA, handles the constraints directly (scipy): `x0`, `fixed`, `rhobeg`, `tol` | box | exactly (replay) |

\* On an indexed pool, a continuous method's proposed point is replaced by
the nearest unused candidate.

**Callback solvers** (`lbfgsb`, `powell`, `bobyqa`, `cobyla`) call the
objective themselves. `continue` resumes them exactly: the solver is rerun
from the same start and answered from the recorded evaluations, at no
cost, until it reaches new points. When one converges it ends the block
early; a later block on that label starts a new run from the best point.
`fixed` (`{"index": value, ...}` in real coordinates) holds those
coordinates at the given values, so the solver moves only the others.

**Warm start** (`"warm_start": true` with `restart`): every earlier
evaluation is told first. Model-based methods (`gp_bo`, `turbo`) fit them.
Population methods (`de`, `pso`) fill their initial population with them,
best first. Start-point methods (`cma_es`, `nelder_mead` and the local
solvers) start at the best of them unless `x0` is given.

**Constraints:** all methods except `cobyla` rank an infeasible point above
every feasible one, by total violation. GP methods cap those values before
fitting.

**Points at a time:** `max_ask` on an optimizer class caps how many points it
can have outstanding; `gp_bo`, `mixed_bo`, `nelder_mead` and the callback
solvers propose one point and wait for its value. `sobol`, `random_search` and
`fixed_points` do not adapt, so a block of theirs is requested all at once.

## Building your own

A custom optimizer may subclass `_base.Base` (`from _base import Base`).
Implement `propose()` (a unit-cube point, an exact candidate index on an
indexed pool, or `None` when finished),
`observe(u, value)` and optionally `absorb(u, value)`. Base handles bounds,
pools, constraint merit, records (`self.U`, `self.y`, `self.best()`) and
pickle save/load.

To use any library solver that calls the objective itself, subclass
`_local.Local` (`from _local import Local`) and implement `solve(f, x0)`:
run the solver on `f` (a unit-cube point of the free coordinates to a
value) from `x0`, with `self.ndim` free coordinates. You get `fixed`,
exact `continue` by replay (the solver must be deterministic given its
inputs; seed any randomness from `self.seed`), and the start point. The
four callback solvers above are short examples. `_gp.fit(U, targets, seed)` fits the GP used
by `gp_bo` and `turbo`.
