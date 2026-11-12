"""A scripted stand-in for the agent, to test the harness without model calls.

Run in the workspace with the harness's prompt on stdin, it does what a call
must: write a proposal and a decision, and validate. Each block is 25% of the
budget (at least 5) of uniform random points, and the blocks alternate between
the two kinds of proposal: an optimizer written to work/, and listed points.
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

OPTIMIZER = '''import pickle
import random


class Optimizer:
    batch = True

    def __init__(self, space, seed=0):
        self.bounds, self.rng = space["bounds"], random.Random(seed)

    def ask(self, n):
        return [[self.rng.uniform(lo, hi) for lo, hi in self.bounds] for _ in range(n)]

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
'''


def main():
    prompt = sys.stdin.read()
    used, budget, remaining = map(int, re.search(r"(\d+) of (\d+) measurements are used; (\d+) remain", prompt).groups())
    project = Path(".agentoptimizer")
    n = min(remaining, max(5, budget // 4))
    block = used // max(5, budget // 4)
    if block % 2 == 0:
        (project / "work" / "random_points.py").write_text(OPTIMIZER)
        proposal = {"script": "work/random_points.py", "label": f"rs{block}", "args": {"seed": block},
                    "n_points": n, "state": "restart"}
    else:
        space = json.loads((project / "space.json").read_text())
        rng = random.Random(block)
        points = [{v["name"]: rng.uniform(v["low"], v["high"]) for v in space["variables"]} for _ in range(n)]
        proposal = {"label": f"listed{block}", "points": points}
    (project / "agent" / "proposal.json").write_text(json.dumps(proposal))
    (project / "agent" / "decision.md").write_text(
        f"**Budget:** {used} used / {budget} total; {remaining} remaining.\n"
        f"**Rationale:** fake agent: {n} uniform random points.\n**Review:** nothing.\n")
    (project / "agent" / "context.md").write_text(f"Fake agent; {used} used.\n")
    return subprocess.run(["skill/aopt", "validate"]).returncode


if __name__ == "__main__":
    sys.exit(main())
