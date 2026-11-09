"""Check an optimizer offline through the same host the runner uses.

    aopt python <skill>/check.py work/my_optimizer.py '{"seed": 0}' [--warm N] [--space PATH]

Runs, on a synthetic objective over the project's encoded space (never the real one):
  1. a new optimizer, told N earlier points first if --warm (as warm_start does),
     then 4 ask/tell rounds and save;
  2. a fresh process that loads the saved state, 4 more rounds, and save;
  3. the way `aopt suggest` works between measurements: ask, save with the point
     outstanding, load in a fresh process, tell, ask again.
Prints OK, or the host's error. State goes to a temporary directory.
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def synthetic(space, x):
    if space["indexed_pool"]:
        x = space["candidates"][x]
    lo, hi = zip(*space["bounds"])
    return sum(((v - l) / (h - l or 1) - 0.3) ** 2 for v, l, h in zip(x, lo, hi))


class Host:
    def __init__(self, script, space_path, state, args, mode):
        self.p = subprocess.Popen([sys.executable, str(ROOT / "host.py"), script, str(space_path),
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


def valid(space, x, used):
    if space["indexed_pool"]:
        assert isinstance(x, int) and x not in used, f"bad or repeated index {x!r}"
        used.add(x)
    else:
        assert len(x) == space["dim"] and all(l <= v <= h for v, (l, h) in zip(x, space["bounds"])), \
            f"point outside bounds: {x}"


def tell(host, space, x):
    host.call({"op": "tell", "X": [x], "y": [synthetic(space, x)], "g": [[0.0] * space["n_constraints"]]})


def rounds(host, space, n, used):
    for _ in range(n):
        X = host.call({"op": "ask", "n": 1})["X"]
        if not X:
            print("optimizer reported it finished")
            return False
        valid(space, X[0], used)
        tell(host, space, X[0])
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("args", nargs="?", default="{}")
    parser.add_argument("--warm", type=int, default=0, help="earlier points told before the first ask")
    parser.add_argument("--space", type=Path, default=Path(".agentoptimizer/space.json"))
    a = parser.parse_args()
    space = json.loads(a.space.read_text())
    state = Path(tempfile.mkdtemp(prefix="aopt_check_")) / "state"
    rng, used = random.Random(0), set()
    host = Host(a.script, a.space, state, json.loads(a.args), "new")
    if a.warm:
        if space["indexed_pool"]:
            X = rng.sample(range(len(space["candidates"])), a.warm)
            used.update(X)
        else:
            X = [[rng.uniform(l, h) for l, h in space["bounds"]] for _ in range(a.warm)]
        host.call({"op": "tell", "X": X, "y": [synthetic(space, x) for x in X],
                   "g": [[0.0] * space["n_constraints"] for _ in X]})
    going = rounds(host, space, 4, used)
    host.call({"op": "save"})
    if going:
        host = Host(a.script, a.space, state, {}, "load")
        going = rounds(host, space, 4, used)
        host.call({"op": "save"})
    if going:
        host = Host(a.script, a.space, state, {}, "load")
        X = host.call({"op": "ask", "n": 1})["X"]
        host.call({"op": "save"})
        if X:
            valid(space, X[0], used)
            host = Host(a.script, a.space, state, {}, "load")
            tell(host, space, X[0])
            nxt = host.call({"op": "ask", "n": 1})["X"]
            host.call({"op": "save"})
            if nxt and nxt[0] == X[0] and not space["noisy"]:
                sys.exit("FAILED\nafter a reload, the optimizer asked again for the point it had just been told")
    print("OK")


if __name__ == "__main__":
    main()
