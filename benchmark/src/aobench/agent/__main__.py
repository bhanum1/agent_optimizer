"""python -m aobench.agent: run the agent harness on benchmark tasks (see aobench/agent/__init__.py)."""
from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..run import FLOOR
from ..tasks import SPECS
from . import REPO, sandbox
from . import __doc__ as DOC
from .campaign import Campaign


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m aobench.agent", description=DOC,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="results and logs (same layout as the baselines)")
    p.add_argument("--work", type=Path, default=Path.home() / ".aobench" / "work",
                   help="agent workspaces; keep this outside the repository")
    p.add_argument("--tasks", nargs="*", default=list(SPECS), choices=list(SPECS))
    p.add_argument("--seeds", type=int, default=1, help="seeds 0..n-1")
    p.add_argument("--agent", default="claude", choices=["claude", "fake"])
    p.add_argument("--model")
    p.add_argument("--effort")
    p.add_argument("--workers", type=int, default=1, help="campaigns run at once")
    p.add_argument("--call-timeout", type=float, default=3600, help="seconds per agent call")
    p.add_argument("--max-calls", type=int, default=200, help="agent calls per campaign")
    a = p.parse_args(argv)
    if REPO in a.work.resolve().parents:
        p.error("--work must be outside the repository, so the agent cannot read the tasks")
    if a.agent == "claude" and not (sandbox.available() and shutil.which("claude")):
        p.error("the claude agent needs `bwrap` (its sandbox, Linux only) and `claude` on PATH")
    jobs = [(t, s) for t in a.tasks for s in range(a.seeds)
            if not (a.out / t / "agent" / f"seed_{s}.csv").exists()]
    print(f"{len(jobs)} campaigns to run", flush=True)

    def one(job):
        t, s = job
        t0 = time.time()
        c = Campaign(t, s, a.out, a.work, a.agent, a.model, a.effort, a.call_timeout, a.max_calls)
        try:
            reason = c.run()
        except Exception as e:
            reason = f"error: {e}"
        score = np.log10(max(c.regret, FLOOR)) if c.rows else float("nan")
        print(f"{t} seed {s}: {reason}, {len(c.rows)}/{c.budget} evaluations, "
              f"log10 regret {score:.2f}, {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, jobs))


if __name__ == "__main__":
    main()
