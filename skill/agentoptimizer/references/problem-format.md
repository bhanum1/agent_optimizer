# problem.yaml

The machine-readable problem definition, in `.agentoptimizer/problem.yaml`.
`aopt init` checks it and reports the first problem it finds.

```yaml
name: catalyst-yield
objective: {name: yield_pct, direction: maximize}   # or minimize (the default)
variables:
  - {name: temperature_c, type: continuous, low: 40, high: 120, resolution: 0.5, unit: C,
     description: "reaction temperature, set on the hot plate"}
  - {name: pressure_bar, type: continuous, low: 0.5, high: 50, log: true}
  - {name: n_cycles, type: integer, low: 1, high: 12}
  - {name: solvent, type: categorical, choices: [water, ethanol, dmso]}
  - {name: stir_rpm, type: ordinal, choices: [200, 400, 800, 1600]}
constraints:
  - {name: impurity_excess, description: "impurity_pct - 2; feasible when <= 0"}
noise: {present: true, replicates_allowed: true}
budget: 60
parallel: 1
seed: 0
```

| Field | Meaning |
|---|---|
| `name` | A short name for the campaign. |
| `objective` | `name` (the data.csv column) and `direction`. |
| `variables` | One entry per setting; see below. At least one. |
| `constraints` | Measured outcomes that must be `<= 0`; each is a data.csv column filled in with the objective. Optional. |
| `noise` | `present`: repeated measurements differ. `replicates_allowed` (default: same as `present`): the agent may request deliberate repeats. `noise: false` is short for no noise. |
| `budget` | Total measurements, including earlier data imported at `init`. Failed runs count; cancelled requests do not. |
| `parallel` | Points requested per round from methods that allow it, for users who run experiments side by side. Default 1. |
| `candidates` | Optional path (relative to `.agentoptimizer/`) of a CSV listing every allowed configuration, one per row, with a column per variable. Optimizers then choose among its rows, each at most once. |
| `seed` | Seed for optimizer randomness. Default 0. |
| `notes`, `known_constraints` | Free text kept with the file; not used by `aopt`. |

## Variables

| Type | Fields | Written in data.csv as |
|---|---|---|
| `continuous` | `low`, `high`; `log: true` to search on a log scale (`low > 0`); `resolution` to round to multiples of it from `low` | a number |
| `integer` | `low`, `high` (whole numbers); `log: true` allowed | a whole number |
| `ordinal` | `choices`: ordered levels, at least two | one of the choices |
| `categorical` | `choices`: unordered options, at least two | one of the choices |

All variables may have `unit` and `description`. Names start with a letter and use
letters, digits, `_`, `.` or `-`; they cannot be `id`, `block`, `label`, `status`,
`replicate_of`, `candidate` or `note`.

## Rules on the inputs

`aopt` handles box bounds and the types above. Rules that tie variables together
(fractions that must sum to 1, combinations that are unsafe or impossible) are not
enforced. Options:
- Reparameterize so the rule holds by construction, e.g. for three fractions
  summing to 1, optimize two of them as shares of the remainder and compute the third.
  Write the mapping in PROBLEM.md so the user knows how to set up each run.
- If the allowed configurations can be listed, make them a candidates file.
- Otherwise, write a custom optimizer in `work/` that only proposes allowed points,
  and record the rule in PROBLEM.md.
