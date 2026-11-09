# How variables are encoded for optimizers

Optimizers work on a numeric box, the encoded space, written to
`.agentoptimizer/space.json`. `aopt` translates between it and the user's values:
each requested point is decoded to a configuration for data.csv, and the optimizer
is told the encoded point it asked for, so its internal state stays consistent.

| Type | Encoded coordinates | Bounds | Decoded by |
|---|---|---|---|
| continuous | 1: the value | `[low, high]` | rounding to `resolution` if given (else to 12 significant digits) |
| continuous, `log` | 1: log10 of the value | `[log10 low, log10 high]` | `10 ** x` |
| integer | 1: a relaxed value | `[low - 0.5, high + 0.5)` | rounding |
| integer, `log` | 1: log10 of a relaxed value | `[log10(low - 0.5), log10(high + 0.5))` | `10 ** x`, then rounding |
| ordinal | 1: the choice index | `[-0.5, k - 0.5)` | rounding to the nearest index |
| categorical | k: one per choice | `[0, 1]` each | the largest coordinate |

A problem whose variables are all linear continuous encodes to exactly its own box,
so points and arguments such as `x0` are in the user's units.

`space.json` holds `dim`, `bounds`, `n_constraints`, `noisy`, `indexed_pool`,
`candidates` (encoded, for a candidates file), `all_continuous`, and `variables`:
for each variable its `name`, `type`, `coords` (its coordinate indices), and its
`low`/`high`/`log` or `choices`. Library optimizers that model types directly
(`mixed_bo`) read `variables`.

To write an encoded point by hand, run `aopt encode '{"temperature_c": 80, ...}'`;
`aopt decode '[...]'` goes the other way. In Python,
`from aoptcore import observations` gives measured points already encoded.

**Repeats.** Several encoded points decode to the same configuration (any point
whose integer coordinates round alike, for example). When an optimizer proposes an
already-measured configuration on a noisy problem, it becomes a replicate; on a
deterministic one, the optimizer is told the recorded value instead and asked again.
After 50 such repeats in a row the block ends early, since the optimizer has
converged on the discrete grid.
