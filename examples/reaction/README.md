# Example: cross-coupling yield

A four-parameter reaction optimization with a simulated lab, for trying
advisor mode (`/opt-run`) without running real experiments.

- `problem.toml`: the campaign definition: temperature, catalyst loading
  (log scale), base equivalents and reaction time; maximize yield; 40
  experiments in blocks of about 8.
- `lab.py`: the simulated lab. It measures every pending experiment with
  noise, fails very hot and long runs, and records the results. It is the
  hidden truth, so keep it out of the advisor's view: work on a copy of the
  campaign outside this folder.

```bash
mkdir -p runs/reaction && cp examples/reaction/problem.toml runs/reaction/
```

Then repeat until the budget is used:

```text
/opt-run runs/reaction                                     # in Claude Code: proposes the next block
.venv/bin/python examples/reaction/lab.py runs/reaction    # "runs" it and records the results
```

The best achievable yield is about 95%.
