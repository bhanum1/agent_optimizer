"""Check an optimizer offline through the same host the runner uses.

    .venv/bin/python check.py work/my_optimizer.py '{"seed": 0}' [--warm N]

Runs, on a synthetic objective (never the real one):
  1. a new optimizer, told N earlier points first if --warm (as warm_start does),
     then 4 ask/tell rounds and save;
  2. a fresh process that loads the saved state, 4 more rounds, and save.
Prints OK, or the host's error. State goes to .tmp/check/.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def synthetic(space, x):
    if space["indexed_pool"]:
        x = space["candidates"][x]
    lo, hi = zip(*space["bounds"])
    return sum(((v - l) / (h - l or 1) - 0.3) ** 2 for v, l, h in zip(x, lo, hi))


class Host:
    def __init__(self, script, state, args, mode):
        self.p = subprocess.Popen([sys.executable, str(ROOT / "host.py"), script, str(ROOT / "space.json"),
                                   str(state), json.dumps(args), mode],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.call(None)

    def call(self, request):
        if request is not None:
            self.p.stdin.write(json.dumps(request) + "\n")
            self.p.stdin.flush()
        line = self.p.stdout.readline()
        reply = json.loads(line) if line else {"error": f"host exited with code {self.p.wait()}"}
        if "error" in reply:
            sys.exit("FAILED\n" + reply["error"])
        return reply


def rounds(host, space, n, used):
    for _ in range(n):
        X = host.call({"op": "ask", "n": 1})["X"]
        if not X:
            print("optimizer reported it finished")
            return
        x = X[0]
        if space["indexed_pool"]:
            assert isinstance(x, int) and x not in used, f"bad or repeated index {x!r}"
            used.add(x)
        else:
            assert len(x) == space["dim"] and all(l <= v <= h for v, (l, h) in zip(x, space["bounds"])), \
                f"point outside bounds: {x}"
        host.call({"op": "tell", "X": [x], "y": [synthetic(space, x)], "g": [[0.0] * space["n_constraints"]]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("args", nargs="?", default="{}")
    parser.add_argument("--warm", type=int, default=0, help="earlier points told before the first ask")
    a = parser.parse_args()
    space = json.loads((ROOT / "space.json").read_text())
    state = ROOT / ".tmp/check/state"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.unlink(missing_ok=True)
    rng, used = random.Random(0), set()
    host = Host(a.script, state, json.loads(a.args), "new")
    if a.warm:
        if space["indexed_pool"]:
            X = rng.sample(range(len(space["candidates"])), a.warm)
            used.update(X)
        else:
            X = [[rng.uniform(l, h) for l, h in space["bounds"]] for _ in range(a.warm)]
        host.call({"op": "tell", "X": X, "y": [synthetic(space, x) for x in X],
                   "g": [[0.0] * space["n_constraints"] for _ in X]})
    rounds(host, space, 4, used)
    host.call({"op": "save"})
    host = Host(a.script, state, {}, "load")
    rounds(host, space, 4, used)
    host.call({"op": "save"})
    print("OK")


if __name__ == "__main__":
    main()
