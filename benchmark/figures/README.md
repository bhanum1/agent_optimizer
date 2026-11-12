# Figures

The benchmark's current figures, kept in git. Each folder has `regret` (mean log₁₀
regret per task, ±1 standard error) and `rank` (average rank over tasks, with each
task's rank behind it), as PDF and PNG.

| Folder | What | Seeds |
|---|---|---|
| `baselines/` | the eight baselines | 0–9 |
| `opus_high/` | the agent on Claude Opus 5.5, effort high, against the baselines | 0–4 |
| `sonnet_medium/` | the agent on Claude Sonnet 5.5, effort medium, against the baselines | 0–2 |

The agent figures use only the seeds the agent ran (`--seeds`), so every method is
compared on the same instances. To redraw them from the run folders:

```bash
.venv/bin/python -m aobench.figures --runs benchmark/runs/main --out benchmark/figures/baselines
.venv/bin/python -m aobench.figures --runs benchmark/runs/main benchmark/runs/opus_high --out benchmark/figures/opus_high --seeds 5
.venv/bin/python -m aobench.figures --runs benchmark/runs/main benchmark/runs/sonnet_medium --out benchmark/figures/sonnet_medium --seeds 3
```

`counterfactual_ill_conditioned_noisy` is a one-off study of Opus high on
`ill_conditioned_noisy`, seeds 0–5. At each block where the agent switched method or
restarted, the campaign was replayed to that point and the optimizer it was using was
run to the end of the budget instead. Each branch is a single noise draw, so
differences under about two decades may be luck. "(approx.)" marks branches after the
replay stopped matching the original run exactly. The script that made it is not in
the repository.
