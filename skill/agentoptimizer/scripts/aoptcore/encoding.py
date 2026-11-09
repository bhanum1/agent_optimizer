"""Translate between a configuration (the user's values) and the optimizer's numeric box.

Each variable becomes one or more coordinates of the encoded space:

    continuous    its value (log10 of it if `log`), bounds [low, high]
    integer       a relaxed value, bounds [low - 0.5, high + 0.5) (log10 if `log`), rounded on decode
    ordinal       the choice index, bounds [-0.5, k - 0.5), rounded on decode
    categorical   one coordinate per choice in [0, 1], decoded by argmax

A problem with only continuous variables encodes to exactly its own box, so
`x0` and other points are given in the user's units. The optimizer is always
told the encoded point it asked for; `canonical` is the encoding of the decoded
configuration, used for warm starts and to detect repeats.
"""
from __future__ import annotations

import math

import numpy as np

from .problem import Problem, Variable

HALF = 0.5 - 1e-9


class Encoder:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.slices: list[slice] = []
        lo, hi = [], []
        for v in problem.variables:
            start = len(lo)
            if v.type == "categorical":
                lo += [0.0] * len(v.choices)
                hi += [1.0] * len(v.choices)
            elif v.type == "ordinal":
                lo.append(-HALF)
                hi.append(len(v.choices) - 1 + HALF)
            else:
                a, b = (math.log10(v.low), math.log10(v.high)) if v.log else (v.low, v.high)
                if v.type == "integer" and not v.log:
                    a, b = a - HALF, b + HALF
                elif v.type == "integer":
                    a, b = math.log10(v.low - HALF), math.log10(v.high + HALF)
                lo.append(a)
                hi.append(b)
            self.slices.append(slice(start, len(lo)))
        self.bounds = np.array([lo, hi]).T
        self.dim = len(lo)
        self.candidates = None
        if problem.candidates is not None:
            from .data import read_candidates
            self.candidate_configs = read_candidates(problem)
            self.candidates = np.array([self.encode(c) for c in self.candidate_configs])

    # ---- the optimizer's view ---------------------------------------------------------------
    def space(self) -> dict:
        """The dict every optimizer is constructed with (also written to space.json)."""
        variables = []
        for v, s in zip(self.problem.variables, self.slices):
            entry = {"name": v.name, "type": v.type, "coords": list(range(s.start, s.stop))}
            if v.type in ("continuous", "integer"):
                entry.update(low=v.low, high=v.high, log=v.log)
            else:
                entry["choices"] = v.choices
            variables.append(entry)
        return {"dim": self.dim, "bounds": self.bounds.tolist(),
                "n_constraints": len(self.problem.constraints), "noisy": self.problem.noisy,
                "indexed_pool": self.candidates is not None,
                "candidates": None if self.candidates is None else self.candidates.tolist(),
                "variables": variables, "all_continuous": self.all_continuous}

    @property
    def all_continuous(self) -> bool:
        return all(v.type == "continuous" and not v.log for v in self.problem.variables)

    # ---- encode / decode ----------------------------------------------------------------------
    def decode(self, x) -> dict:
        """An encoded point (or candidate index) -> a configuration {name: value}."""
        if self.candidates is not None and isinstance(x, (int, np.integer)):
            return dict(self.candidate_configs[int(x)])
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,) or not np.isfinite(x).all():
            raise ValueError(f"expected a finite point with {self.dim} encoded coordinates, got {x.tolist()}")
        x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        return {v.name: _decode(v, x[s]) for v, s in zip(self.problem.variables, self.slices)}

    def encode(self, config: dict) -> list[float]:
        """A configuration -> its canonical encoded point."""
        out = []
        for v in self.problem.variables:
            value = config[v.name]
            if v.type == "categorical":
                one_hot = [0.0] * len(v.choices)
                one_hot[choice_index(v, value)] = 1.0
                out += one_hot
            elif v.type == "ordinal":
                out.append(float(choice_index(v, value)))
            else:
                value = float(value)
                out.append(math.log10(value) if v.log else value)
        return out

    def canonical(self, x) -> list[float]:
        return self.encode(self.decode(x))

    def key(self, config: dict) -> tuple:
        """A hashable identity of a configuration, for detecting repeats."""
        return tuple(format_value(v, config[v.name]) for v in self.problem.variables)

    def candidate_index(self, config: dict) -> int | None:
        if self.candidates is None:
            return None
        k = self.key(config)
        for i, c in enumerate(self.candidate_configs):
            if self.key(c) == k:
                return i
        return None


def _decode(v: Variable, x):
    if v.type == "categorical":
        return v.choices[int(np.argmax(x))]
    if v.type == "ordinal":
        return v.choices[int(np.clip(round(float(x[0])), 0, len(v.choices) - 1))]
    value = 10 ** float(x[0]) if v.log else float(x[0])
    if v.type == "integer":
        return int(np.clip(round(value), v.low, v.high))
    if v.resolution:
        steps = round((value - v.low) / v.resolution)
        value = v.low + steps * v.resolution
        digits = max(0, -math.floor(math.log10(v.resolution)) + 6)
        value = round(value, digits)
    else:
        value = float(f"{value:.12g}")  # drop float noise, e.g. from the log round trip
    return float(np.clip(value, v.low, v.high))


def choice_index(v: Variable, value) -> int:
    text = str(value).strip()
    for i, c in enumerate(v.choices):
        if str(c) == text:
            return i
    for i, c in enumerate(v.choices):  # numeric choices written differently, e.g. 400 vs 400.0
        try:
            if float(c) == float(text):
                return i
        except (TypeError, ValueError):
            pass
    raise ValueError(f"{v.name}: {value!r} is not one of {v.choices}")


def format_value(v: Variable, value) -> str:
    """How a variable's value is written in data.csv."""
    if v.type in ("categorical", "ordinal"):
        return str(v.choices[choice_index(v, value)])
    if v.type == "integer":
        return str(int(round(float(value))))
    return repr(float(value))


def parse_value(v: Variable, text: str):
    """A data.csv cell -> the variable's value; raises ValueError if invalid or out of range."""
    text = str(text).strip()
    if v.type in ("categorical", "ordinal"):
        return v.choices[choice_index(v, text)]
    value = float(text)
    if v.type == "integer":
        if value != int(value):
            raise ValueError(f"{v.name}: {text} is not a whole number")
        value = int(value)
    if not math.isfinite(value) or not v.low <= value <= v.high:
        raise ValueError(f"{v.name}: {text} is outside [{v.low:g}, {v.high:g}]")
    return value
