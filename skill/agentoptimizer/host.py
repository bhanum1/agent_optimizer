"""Run one optimizer for one block, speaking JSON lines on stdin/stdout.

Runs inside the sandbox and never sees the objective. Standalone: it
imports nothing from this repository, so it can be copied into a workspace.

    python host.py OPTIMIZER_FILE SPACE_JSON STATE_PATH ARGS_JSON (new|load)

OPTIMIZER_FILE defines a class `Optimizer`:

    Optimizer(space, **args)       space: dict from SPACE_JSON
    ask(n) -> list                 n points (lists of floats), or n candidate
                                   indices for an indexed-pool space; an empty
                                   list means the optimizer has finished
    tell(X, y, g) -> None          X as asked; y floats; g one list per point
    save(path) -> None
    Optimizer.load(path)           classmethod returning the saved optimizer

Requests, one JSON object per line:
    {"op": "ask", "n": k}                   -> {"X": [...]}
    {"op": "tell", "X": .., "y": .., "g": ..} -> {}
    {"op": "save"}                          -> {}   (then the host exits)
Any exception is reported as {"error": "<traceback>"} and the host exits 1.
Stdout is reserved for replies; the optimizer's own prints go to stderr.
"""

import importlib.util
import json
import sys
import traceback
from pathlib import Path

LIB = Path(__file__).resolve().parent / "lib"  # library modules may import each other


def load_class(path):
    # Registered under a fixed name, so pickled state from one block loads in the next.
    spec = importlib.util.spec_from_file_location("block_optimizer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["block_optimizer"] = module
    if str(LIB) not in sys.path:
        sys.path.insert(0, str(LIB))
    spec.loader.exec_module(module)
    return module.Optimizer


def main(argv):
    path, space_path, state_path, args, mode = argv
    replies = sys.stdout
    sys.stdout = sys.stderr
    try:
        cls = load_class(path)
        with open(space_path) as f:
            space = json.load(f)
        opt = cls.load(state_path) if mode == "load" else cls(space, **json.loads(args))
        print(json.dumps({}), file=replies, flush=True)  # ready
        for line in sys.stdin:
            request = json.loads(line)
            op = request["op"]
            if op == "ask":
                reply = {"X": [_plain(x) for x in opt.ask(request["n"])]}
            elif op == "tell":
                opt.tell(request["X"], request["y"], request["g"])
                reply = {}
            elif op == "save":
                opt.save(state_path)
                print(json.dumps({}), file=replies, flush=True)
                return 0
            else:
                raise ValueError(f"unknown op {op!r}")
            print(json.dumps(reply), file=replies, flush=True)
        raise RuntimeError("stdin closed before save")
    except Exception:
        print(json.dumps({"error": traceback.format_exc()}), file=replies, flush=True)
        return 1


def _plain(x):
    """A point as a list of floats, or a candidate index as an int."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    return x if isinstance(x, int) else [float(v) for v in x]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
