"""Base for local solvers whose libraries call f themselves (callback style).

The solver runs in a thread and each call to f becomes one ask/tell. A
running solver cannot be pickled, so the state saved between blocks is the
settings, the start point and the values of every call so far. `continue`
resumes exactly by replay: the solver restarts from the same start and the
recorded calls are answered from that record, costing no evaluations, until
it reaches new points. This needs a solver that is deterministic given its
inputs (seeded randomness is fine); if a replayed point differs from the
record, the run restarts fresh from the best point instead.

When the solver finishes (converges), the block ends early; a later block
starts a new run from the best point.

`fixed` ({index: value} in real coordinates) holds those coordinates at
the given values; the solver then works only in the remaining ones
(`self.ndim` of them).

A subclass implements solve(f, x0): run the solver on f (a point of the
free coordinates in the unit cube -> value) from x0, using `self.ndim`.
"""
import queue
import sys
import threading

import numpy as np

from _base import Base


class Local(Base):
    max_ask = 1  # the solver waits for each value before its next call
    def __init__(self, space, seed=0, x0=None, fixed=None):
        super().__init__(space, seed)
        if self.indexed:
            raise ValueError("local solvers need a continuous space, not an indexed pool")
        fixed = {int(k): float(v) for k, v in (fixed or {}).items()}
        if any(not 0 <= k < self.dim for k in fixed):
            raise ValueError(f"fixed indices must be in 0..{self.dim - 1}")
        self.fixed_x = fixed
        self.fixed_u = {k: float(self.to_unit(np.full(self.dim, v))[k]) for k, v in fixed.items()}
        self.free = np.array([i for i in range(self.dim) if i not in fixed], dtype=int)
        if not len(self.free):
            raise ValueError("fixed leaves no free coordinates")
        self.ndim = len(self.free)
        self.x0 = None if x0 is None else self.to_unit(x0)
        self.thread, self.starts = None, 0
        self.run_x0 = None   # free-coordinate start of the current run
        self.calls = []      # (point, value, raw (y, g)) for every call of the current run
        self.last = None     # raw (y, g) of the latest observation, for constrained solvers

    def solve(self, f, x0):
        """Run the solver on f (free-coordinate unit point -> value) from x0."""
        raise NotImplementedError

    # ---- coordinates -----------------------------------------------------------
    def full(self, v):
        u = np.empty(self.dim)
        u[self.free] = np.clip(np.asarray(v, dtype=float), 0, 1)
        for k, val in self.fixed_u.items():
            u[k] = val
        return u

    def from_unit(self, u):
        x = super().from_unit(u)
        for k, v in self.fixed_x.items():  # exact, not round-tripped through the unit cube
            x[k] = v
        return x

    def start_point(self):
        """The best point told so far that matches the fixed coordinates (else any), as free coordinates."""
        if self.x0 is not None and not self.starts:
            return self.x0[self.free]
        if not len(self.y):
            return np.full(self.ndim, 0.5)
        y = self.y.copy()
        if self.fixed_u:
            match = np.all([np.isclose(self.U[:, k], v, atol=1e-9) for k, v in self.fixed_u.items()], axis=0)
            if match.any():
                y[~match] = np.inf
        return self.U[int(np.argmin(y)), self.free]

    # ---- solver thread ---------------------------------------------------------
    def launch(self, x0, replay):
        self.asks, self.answers = queue.Queue(), queue.Queue()
        record = list(self.calls) if replay else []
        if not replay:
            self.calls, self.run_x0 = [], np.asarray(x0, dtype=float)
        position = [0]

        def f(v):
            v = np.clip(np.asarray(v, dtype=float), 0, 1)
            i = position[0]
            position[0] += 1
            if i < len(record):
                point, value, raw = record[i]
                if not np.allclose(point, v, rtol=0, atol=1e-12):
                    raise Diverged(i)
                self.last = raw
                return value
            self.asks.put(v)
            return self.answers.get()  # observe() records the call before answering

        def run():
            np.random.seed(self.seed)  # for solvers that draw from numpy's global generator
            try:
                self.solve(f, np.clip(self.run_x0, 0, 1))
                self.asks.put(None)
            except BaseException as e:  # surface solver errors in the host
                self.asks.put(e)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def next_ask(self):
        item = self.asks.get()
        if isinstance(item, Diverged):
            print(f"Replay diverged at call {item.args[0]}; restarting from the best point.", file=sys.stderr)
            self.starts += 1
            self.launch(self.start_point(), replay=False)
            return self.next_ask()
        if isinstance(item, BaseException):
            raise item
        if item is None:
            self.thread, self.calls, self.run_x0 = "finished", [], None
        self.current = item
        return item

    def propose(self):
        if self.thread == "finished":
            return None
        if self.thread is None:
            if self.run_x0 is not None:
                self.launch(self.run_x0, replay=True)
            else:
                x0 = self.start_point()
                self.starts += 1
                self.launch(x0, replay=False)
        v = self.next_ask()
        return None if v is None else self.full(v)

    def tell(self, X, y, g):
        for x, yi, gi in zip(X, y, g):
            self.last = (float(yi), [float(v) for v in gi])
            awaiting = self.thread is None and self.run_x0 is not None and getattr(self, "current", None) is not None
            if awaiting and tuple(float(v) for v in x) not in self.pending and \
                    np.allclose(self.to_unit(x)[self.free], self.current, rtol=0, atol=1e-9):
                # Loaded from a save made while this point was outstanding: record it, so the
                # replay answers it from the record instead of asking for it again.
                super().tell([x], [yi], [gi])
                self.calls.append((self.current, float(self.y[-1]), self.last))
                self.current = None
                continue
            super().tell([x], [yi], [gi])

    def observe(self, u, value):
        # Recorded here, in the caller's thread, so a save right after tell() always includes it.
        self.calls.append((self.current, value, self.last))
        self.answers.put(value)

    def __getstate__(self):
        state = dict(self.__dict__, thread=None)  # a finished run restarts from the best point next block
        state.pop("asks", None)
        state.pop("answers", None)
        state["pending"] = {}  # an asked, untold point is re-asked by the replay
        return state


class Diverged(Exception):
    """A replayed solver asked for a point that differs from the record."""
