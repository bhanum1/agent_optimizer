# Task descriptions and recorded budgets

These descriptions use the supplied task descriptions and task registry. Missing units or anonymous identities are not inferred. Budgets are taken from the exported historical records. Candidate indices refer to zero-based data rows, not molecule IDs.

| Task | Candidates | Target | Protocol / notes | Recorded budgets |
| --- | ---: | --- | --- | ---: |
| [btch](tasks/btch/) | 12,708 | Maximize measured C2 yield in oxidative coupling of methane (%) | Batch; at most 20 rows sharing x51–x54 | 1000 |
| [mat_1](tasks/mat_1/) | 9,525 | Maximize simulated MOF CO2 uptake, 298 K and 0.15 bar | Scalar; units not explicitly stated in description.txt | 250 |
| [mat_2](tasks/mat_2/) | 7,525 | Maximize simulated MOF CO2 uptake, 298 K and 16 bar | Scalar; units not explicitly stated in description.txt | 250 |
| [mat_3](tasks/mat_3/) | 10,987 | Maximize log10 of calculated VRH shear modulus in GPa | Scalar; target is already logarithmic | 250 |
| [mat_5](tasks/mat_5/) | 8,540 | Maximize simulated MOF CO2 uptake, 298 K and 0.15 bar (mol/kg) | Scalar | 100 |
| [mat_6](tasks/mat_6/) | 8,540 | Maximize simulated MOF CO2 uptake, 298 K and 16 bar (mol/kg) | Scalar | 100 |
| [mix_1](tasks/mix_1/) | 5,595 | Minimize binary-mixture viscosity at room temperature (cP) | Scalar | 250 |
| [mol_1](tasks/mol_1/) | 848 | Maximize source y in an anonymized molecular pool | Scalar; no description.txt supplied; units unspecified | 100 |
| [mol_2](tasks/mol_2/) | 3,439 | Maximize source y in an anonymized molecular pool | Scalar; no description.txt supplied; units unspecified | 250 |
| [mol_3](tasks/mol_3/) | 1,593 | Minimize substance viscosity at room temperature (cP) | Scalar | 250 |
| [ptr](tasks/ptr/) | 149,361 | Maximize source y in the supplied finite pool | Unrestricted batch; at most 50 candidates; units unspecified | 1000 |
| [rct_1](tasks/rct_1/) | 3,696 | Maximize measured palladium-catalyzed coupling yield (%) | Scalar; 100 degrees Celsius, one minute | 250 |
| [rct_2](tasks/rct_2/) | 1,728 | Maximize reaction yield (%) and minimize supplied reaction cost | Multiobjective; one query reveals both labels | 250 |
| [sup](tasks/sup/) | 21,263 | Maximize superconducting critical temperature (K) | Scalar | 250 |

## Source data and context

Task tables are under [`../data/`](../data/). Where available, `description.txt` supplies the domain description and `meta.csv`/`id.csv` supply additional descriptors or identities. Full-pool outcomes and identities belong to the trusted evaluator and offline scoring; they must not be supplied as unqueried feedback to an optimization agent.

The same task can appear under several model, effort and information conditions. Use each seed’s `metadata.json` for the recorded model and effort, and inspect its original prompt/decision/context files in [`agent_reasoning/`](agent_reasoning/README.md) to see what information was actually supplied. Domain-context, representation-design and anonymous runs are distinct experimental conditions.

## Protocol and budget caveats

`btch` uses 1,000 candidate evaluations, grouped by identical x51–x54 conditions with up to 20 rows per synchronous batch. `ptr` results here use the historical 1,000-evaluation comparison (20 batches of 50 for full batches); they are not the separate 5,000-evaluation campaign. `rct_2` uses 250 paired candidate queries, revealing yield and cost together. These tasks remain outside the scalar protocol.

All other listed completed records use their saved scalar budgets. Exporting results does not rerun or change historical campaign budgets. Additional failed, incomplete and preflight agent attempts are preserved only in the reasoning export and may have fewer evaluations or different test budgets. Missing method/seed combinations are not manufactured.
