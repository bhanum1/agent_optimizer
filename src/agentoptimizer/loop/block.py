"""Run one block: drive a sandboxed optimizer host for exactly n_points evaluations.

The objective is evaluated here, in the runner. The host only proposes
points and receives values, so a block can never exceed its length.
"""

from __future__ import annotations

import json
import selectors
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HOST = Path(__file__).with_name("host.py")


@dataclass
class Block:
    number: int
    label: str          # optimizer name recorded in data.csv
    script: Path        # file defining class Optimizer
    args: dict
    n_points: int
    state: str          # "continue" (load saved state) or "restart" (construct new)
    warm_start: bool    # on restart, tell the new optimizer every earlier evaluation


@dataclass
class Result:
    used: int
    error: str | None   # None if the block completed and its state was saved
    state_saved: bool
    finished: bool = False  # the optimizer reported it had nothing more to propose


class HostError(Exception):
    pass


class Host:
    def __init__(self, command: list[str], timeout: float, log: Path, env: dict | None):
        self.timeout, self.log = timeout, log
        with open(log, "a") as stderr:  # a file, so a chatty optimizer cannot fill a pipe
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=stderr, text=True, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def call(self, request: dict | None) -> dict:
        if request is not None:
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                raise HostError(self._exited())
        if not self.selector.select(self.timeout):
            raise HostError(f"no reply within {self.timeout:.0f} s")
        line = self.process.stdout.readline()
        if not line:
            raise HostError(self._exited())
        reply = json.loads(line)
        if "error" in reply:
            raise HostError(reply["error"])
        return reply

    def _exited(self) -> str:
        self.process.wait(5)
        return f"host exited with code {self.process.returncode}\n{self.log.read_text()[-4000:]}"

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.selector.close()
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except BrokenPipeError:
                pass


def run(task, store, block: Block, state_path: Path, space_path: Path, log: Path | None = None,
        prefix: list[str] = (), python: str = sys.executable, env: dict | None = None,
        timeout: float = 600, host_script: Path = HOST) -> Result:
    """Run `block` against `task`, appending every evaluation to `store`.
    `prefix` wraps the host command (the sandbox launcher); the host's stderr goes to `log`."""
    mode = "load" if block.state == "continue" else "new"
    command = [*prefix, python, str(host_script), str(block.script), str(space_path),
               str(state_path), json.dumps(block.args), mode]
    used = 0
    host = Host(command, timeout, log or state_path.with_name(f"block_{block.number}.log"), env)
    try:
        host.call(None)  # ready
        if mode == "new" and block.warm_start and len(store):
            X, y, g = store.history()
            host.call({"op": "tell", "X": X, "y": y, "g": g})
        finished = False
        while used < block.n_points:
            x = host.call({"op": "ask", "n": 1})["X"]
            if not x:
                finished = True
                break
            if len(x) != 1:
                raise HostError(f"ask(1) returned {len(x)} items")
            x, index = _check(task, store, x[0])
            ev = task.evaluate_index(index) if index is not None else task.evaluate(x)
            store.append(block.number, block.label, x, ev, index)
            used += 1
            host.call({"op": "tell", "X": [index if index is not None else list(x)],
                       "y": [ev.y], "g": [list(ev.g)]})
        host.call({"op": "save"})
        return Result(used, None, True, finished)
    except (HostError, ValueError, json.JSONDecodeError) as e:
        return Result(used, str(e), False)
    finally:
        host.close()


def _check(task, store, x):
    """Validate one proposal; return (point, candidate index or None)."""
    if getattr(task, "indexed_pool", False):
        if not isinstance(x, int) or not 0 <= x < task.n_candidates:
            raise ValueError(f"proposal {x!r} is not a candidate index in [0, {task.n_candidates})")
        if x in store.used_indices():
            raise ValueError(f"candidate {x} was already evaluated")
        return np.asarray(task.candidates[x], dtype=float), x
    x = np.asarray(x, dtype=float)
    bounds = np.asarray(task.bounds, dtype=float)
    if x.shape != (task.dim,) or not np.isfinite(x).all():
        raise ValueError(f"proposal is not a finite point of dimension {task.dim}")
    if (x < bounds[:, 0]).any() or (x > bounds[:, 1]).any():
        raise ValueError("proposal is outside the bounds")
    return x, None
