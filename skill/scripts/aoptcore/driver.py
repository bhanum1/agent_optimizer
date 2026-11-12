"""Work through a block with its optimizer: tell it new results, ask it for the next points.

A `Session` opens the current block's optimizer, tells it every result of the
block it has not seen, asks it for the next point(s), writes them to data.csv
as pending rows, and saves its state. `aopt suggest` runs one session per call,
so the optimizer is saved and reloaded between measurements; the benchmark keeps
one session open for a whole block.

The optimizer runs through a handle: `InProcess` (the CLI) or `HostProcess`
(the benchmark's sandboxed host.py, same JSON-lines protocol).
"""
from __future__ import annotations

import importlib.util
import json
import selectors
import subprocess
import sys
from pathlib import Path

from .project import SKILL, Project

MAX_REPEATS = 50  # consecutive already-measured proposals before a block is treated as finished


class HandleError(RuntimeError):
    pass


# ---- optimizer handles --------------------------------------------------------------------------
def _host_module():
    spec = importlib.util.spec_from_file_location("aopt_host", SKILL / "host.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InProcess:
    def __init__(self, script: Path, space: dict, state: Path, args: dict, mode: str):
        self.state = state
        try:
            cls = _host_module().load_class(str(script))
            self.opt = cls.load(str(state)) if mode == "load" else cls(space, **args)
        except Exception as e:
            raise HandleError(f"{type(e).__name__}: {e}") from e
        self.max_ask = getattr(self.opt, "max_ask", None)
        self.batch = bool(getattr(self.opt, "batch", False))

    def ask(self, n):
        try:
            return [_plain(x) for x in self.opt.ask(n)]
        except Exception as e:
            raise HandleError(f"{type(e).__name__}: {e}") from e

    def tell(self, X, y, g):
        try:
            self.opt.tell(X, y, g)
        except Exception as e:
            raise HandleError(f"{type(e).__name__}: {e}") from e

    def save(self):
        self.opt.save(str(self.state))

    def close(self):
        pass


class HostProcess:
    """host.py in a subprocess (e.g. under a sandbox prefix), speaking JSON lines."""

    def __init__(self, script: Path, space_path: Path, state: Path, args: dict, mode: str, prefix=(),
                 python: str = sys.executable, env: dict | None = None, log: Path | None = None,
                 timeout: float = 600, host: Path = SKILL / "host.py"):
        command = [*prefix, python, str(host), str(script), str(space_path), str(state), json.dumps(args), mode]
        self.timeout, self.log = timeout, Path(log or state.with_suffix(".log"))
        with open(self.log, "a") as stderr:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=stderr, text=True, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        ready = self.call(None)
        self.max_ask, self.batch = ready.get("max_ask"), ready.get("batch", False)

    def call(self, request):
        if request is not None:
            try:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                raise HandleError(self._exited())
        if not self.selector.select(self.timeout):
            raise HandleError(f"no reply within {self.timeout:.0f} s")
        line = self.process.stdout.readline()
        if not line:
            raise HandleError(self._exited())
        reply = json.loads(line)
        if "error" in reply:
            raise HandleError(reply["error"])
        return reply

    def _exited(self):
        self.process.wait(5)
        return f"host exited with code {self.process.returncode}\n{self.log.read_text()[-4000:]}"

    def ask(self, n):
        return self.call({"op": "ask", "n": n})["X"]

    def tell(self, X, y, g):
        self.call({"op": "tell", "X": X, "y": y, "g": g})

    def save(self):
        self.call({"op": "save"})

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.selector.close()
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except BrokenPipeError:
                pass


def _plain(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    return int(x) if isinstance(x, int) else [float(v) for v in x]


# ---- sessions -----------------------------------------------------------------------------------
class Session:
    def __init__(self, project: Project, block: dict | None = None, handle=None):
        """`handle(script, space, space_path, state_path, args, mode)` builds the optimizer handle."""
        self.project = project
        self.block = block or project.current()
        if self.block is None:
            raise HandleError("No block yet: the agent proposes one with `aopt commit`.")
        self.make = handle or (lambda script, space, space_path, state, args, mode:
                               InProcess(script, space, state, args, mode))
        self.handle = None
        self.st: dict = {}
        self.messages: list[str] = []

    @property
    def label(self) -> str:
        return self.block["label"]

    def open(self) -> None:
        p, b = self.project, self.block
        st = p.label_state(self.label)
        script = p.root / b["source_file"]
        space = p.encoder.space()
        if st.get("block") == b["block"]:
            if not st.get("saved"):
                raise HandleError(f"The optimizer for block {b['block']} crashed and its state was lost. "
                                  "Ask the agent for a new block (state restart).")
            self.st = st
            self.handle = self.make(script, space, p.space_path, p.state_path(self.label), {}, "load")
            return
        mode = "load" if b["state"] == "continue" else "new"
        self.handle = self.make(script, space, p.space_path, p.state_path(self.label), b["args"], mode)
        self.st = {"block": b["block"], "source_file": b["source_file"], "asked": {}, "told": [],
                   "aliases": {}, "saved": False, "finished": False}
        if mode == "new" and b["warm_start"]:
            rows = [r for r in p.table.rows if int(r["block"]) < b["block"] and r["status"] in ("measured", "failed")]
            if rows:
                self.handle.tell(*self._results(rows, canonical=True))
                self.st["told"] += [r["id"] for r in rows]

    def _results(self, rows, canonical=False):
        t, enc = self.project.table, self.project.encoder
        X, y, g = [], [], []
        penalty = t.penalty()
        for r in rows:
            if canonical:
                X.append(int(r["candidate"]) if enc.candidates is not None else enc.encode(t.config(r)))
            else:
                X.append(self.st["asked"][r["id"]])
            if r["status"] == "measured":
                y.append(t.value(r))
                g.append(t.g(r))
            else:  # failed: worse than everything measured
                y.append(penalty)
                g.append([0.0] * len(self.project.problem.constraints))
        return X, y, g

    def sync(self) -> int:
        """Tell the optimizer every result of this block it has not been told. Returns how many."""
        told = set(self.st["told"])
        rows = [r for r in self.project.table.block_rows(self.block["block"])
                if r["status"] in ("measured", "failed") and r["id"] in self.st["asked"] and r["id"] not in told]
        if rows:
            X, y, g = self._results(rows)
            for r, yi, gi in zip(rows, y, g):  # repeats of a pending point get the same result
                for x in self.st["aliases"].pop(r["id"], []):
                    X.append(x), y.append(yi), g.append(gi)
            self.handle.tell(X, y, g)
            self.st["told"] += [r["id"] for r in rows]
        return len(rows)

    def propose(self) -> list[dict]:
        """Ask for the next point(s) and add them to data.csv as pending rows."""
        p, t, enc = self.project, self.project.table, self.project.encoder
        progress = p.block_progress(self.block)
        if self.st.get("finished") or progress["left"] <= 0:
            return []
        per_round = progress["left"] if self.handle.batch else p.problem.parallel
        want = min(progress["left"], per_round - len(progress["pending"]))
        if self.handle.max_ask:
            want = min(want, self.handle.max_ask - len(progress["pending"]))
        new, repeats = [], 0
        while want > 0:
            X = self.handle.ask(1)
            if not X:
                self.st["finished"] = True
                self.messages.append("The optimizer reported it has finished (converged or exhausted); "
                                     "the block ends early.")
                break
            x = X[0]
            config = enc.decode(x)
            measured = t.find(config, ("measured",))
            pending = t.find(config, ("pending",))
            if p.problem.replicates and (measured or pending):
                r = t.add(self.block["block"], self.label, config, replicate_of=(measured or pending)["id"])
            elif measured:  # deterministic: no need to measure it again
                self.handle.tell([x], [t.value(measured)], [t.g(measured)])
                repeats += 1
                if repeats >= MAX_REPEATS:
                    self.st["finished"] = True
                    self.messages.append(f"The optimizer proposed {MAX_REPEATS} already-measured configurations "
                                         "in a row; the block ends early.")
                    break
                continue
            elif pending:  # deterministic: wait for the pending measurement, then tell both
                self.st["aliases"].setdefault(pending["id"], []).append(x)
                break
            else:
                r = t.add(self.block["block"], self.label, config)
            self.st["asked"][r["id"]] = x
            new.append(r)
            want -= 1
        return new

    def close(self, save: bool = True) -> None:
        try:
            if save and self.handle is not None:
                self.handle.save()
                self.st["saved"] = True
        except HandleError:
            self.st["saved"] = False
            raise
        finally:
            if self.handle is not None:
                self.handle.close()
            self.project.save_label_state(self.label, self.st)
            self.project.table.save()

    def fail(self, error: str) -> None:
        """Record a crashed optimizer: its state is lost, the block stops."""
        self.st["saved"] = False
        self.st["finished"] = True
        if self.handle is not None:
            self.handle.close()
        self.project.save_label_state(self.label, self.st)
        self.project.table.save()
        self.project.note(f"The optimizer of block {self.block['block']} ({self.label}) crashed; its state "
                          f"was not saved, so a later block for this label must restart.\n\n```\n{error[-3000:]}\n```")


def suggest(project: Project) -> tuple[list[dict], list[str]]:
    """One round: tell the current block's optimizer its new results, then ask for the next point(s)."""
    s = Session(project)
    try:
        s.open()
        s.sync()
        rows = s.propose()
    except HandleError as e:
        s.fail(str(e))
        raise
    s.close()
    return rows, s.messages


def settle(project: Project) -> None:
    """Tell the current block's optimizer any results it has not seen, without asking for more."""
    block = project.current()
    if block is None:
        return
    st = project.label_state(block["label"])
    if st.get("block") != block["block"] or not st.get("saved"):
        return
    told = set(st["told"])
    if not any(r["id"] in st["asked"] and r["id"] not in told and r["status"] in ("measured", "failed")
               for r in project.table.rows):
        return
    s = Session(project, block)
    try:
        s.open()
        s.sync()
    except HandleError as e:
        s.fail(str(e))
        return
    s.close()
