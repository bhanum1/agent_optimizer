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

To run it unattended from a terminal (each block is a fresh headless
session, as in the test below):

```bash
for block in 1 2 3 4 5; do
  claude -p "/opt-run runs/reaction" --permission-mode acceptEdits
  .venv/bin/python examples/reaction/lab.py runs/reaction
done
```

On 2026-11-02 five blocks this way (Claude Opus 5.5) went Sobol → batch
GP-BO → trust-region GP-BO → recentred, with a replicate → refine, and
reached 93.0% after 38 of the 40 experiments (one failed in the tar
region), for about $1.90 in model calls. Every
block's `propose.py` reproduced its proposals exactly from the data it saw.
With the `agentoptimizer.batch` proposers (Sobol → qLogNEI → trust region),
the same test cost $1.57, each script was 6–7 lines, and it reached 92.2%
by block 3.
