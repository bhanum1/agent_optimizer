"""problem.yaml: the machine-readable problem definition.

    name: catalyst-yield
    objective: {name: yield_pct, direction: maximize}
    variables:
      - {name: temperature_c, type: continuous, low: 40, high: 120, resolution: 0.5}
      - {name: pressure_bar, type: continuous, low: 0.5, high: 50, log: true}
      - {name: n_cycles, type: integer, low: 1, high: 12}
      - {name: solvent, type: categorical, choices: [water, ethanol, dmso]}
      - {name: stir_rpm, type: ordinal, choices: [200, 400, 800, 1600]}
    constraints:                  # measured with the objective; feasible iff value <= 0
      - {name: impurity_excess}
    noise: {present: true}        # replicates_allowed defaults to present
    budget: 60                    # measurements in total, including the initial data
    parallel: 1                   # optional: points requested per round
    candidates: candidates.csv    # optional: a finite pool, one row per allowed configuration
    seed: 0                       # optional

See references/problem-format.md in the skill folder for every field.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{0,63}")
RESERVED = {"id", "block", "label", "status", "replicate_of", "candidate", "note"}
TYPES = ("continuous", "integer", "categorical", "ordinal")


class ProblemError(ValueError):
    pass


@dataclass
class Variable:
    name: str
    type: str
    low: float | None = None
    high: float | None = None
    log: bool = False
    resolution: float | None = None
    choices: list = field(default_factory=list)
    unit: str = ""
    description: str = ""


@dataclass
class Problem:
    name: str
    objective: str
    maximize: bool
    variables: list[Variable]
    constraints: list[str]
    noisy: bool
    replicates: bool
    budget: int
    parallel: int = 1
    candidates: Path | None = None
    seed: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        """Variable, objective and constraint columns of data.csv, in order."""
        return [v.name for v in self.variables] + [self.objective] + self.constraints


def load(path: Path) -> Problem:
    import yaml
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ProblemError(f"{path.name} is not valid YAML: {e}")
    return parse(raw, path.parent)


def parse(raw, base: Path = Path(".")) -> Problem:
    if not isinstance(raw, dict):
        raise ProblemError("problem.yaml must be a mapping.")
    unknown = set(raw) - {"name", "objective", "variables", "constraints", "noise", "budget",
                          "parallel", "candidates", "seed", "known_constraints", "notes"}
    if unknown:
        raise ProblemError(f"unknown fields {sorted(unknown)}.")
    objective = raw.get("objective")
    if isinstance(objective, str):
        objective = {"name": objective}
    if not isinstance(objective, dict) or "name" not in objective:
        raise ProblemError("objective needs a name, e.g. {name: yield_pct, direction: maximize}.")
    direction = objective.get("direction", "minimize")
    if direction not in ("minimize", "maximize"):
        raise ProblemError("objective.direction must be minimize or maximize.")
    variables = [_variable(v, i) for i, v in enumerate(raw.get("variables") or [])]
    if not variables:
        raise ProblemError("variables must list at least one variable.")
    constraints = []
    for c in raw.get("constraints") or []:
        constraints.append(c["name"] if isinstance(c, dict) else c)
    names = [v.name for v in variables] + [objective["name"]] + constraints
    for n in names:
        if not isinstance(n, str) or not NAME.fullmatch(n) or n in RESERVED:
            raise ProblemError(f"{n!r} is not a usable column name: start with a letter, use letters, digits, "
                               f"_ . - only, and avoid {sorted(RESERVED)}.")
    if len(set(names)) != len(names):
        raise ProblemError("variable, objective and constraint names must all differ.")
    noise = raw.get("noise") or {}
    if isinstance(noise, bool):
        noise = {"present": noise}
    noisy = bool(noise.get("present", False))
    budget = raw.get("budget")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ProblemError("budget must be a positive integer (total measurements, initial data included).")
    parallel = raw.get("parallel", 1)
    if not isinstance(parallel, int) or parallel < 1:
        raise ProblemError("parallel must be a positive integer.")
    candidates = raw.get("candidates")
    if candidates is not None:
        candidates = Path(candidates)
        candidates = candidates if candidates.is_absolute() else Path(base) / candidates
        if not candidates.is_file():
            raise ProblemError(f"candidates file {candidates} does not exist.")
    return Problem(name=str(raw.get("name", "problem")), objective=objective["name"],
                   maximize=direction == "maximize", variables=variables, constraints=constraints,
                   noisy=noisy, replicates=bool(noise.get("replicates_allowed", noisy)), budget=budget,
                   parallel=parallel, candidates=candidates, seed=int(raw.get("seed", 0)), raw=raw)


def _variable(v, i) -> Variable:
    if not isinstance(v, dict):
        raise ProblemError(f"variable {i} must be a mapping.")
    unknown = set(v) - {"name", "type", "low", "high", "log", "resolution", "choices", "unit", "description"}
    if unknown:
        raise ProblemError(f"variable {v.get('name', i)}: unknown fields {sorted(unknown)}.")
    name, kind = v.get("name"), v.get("type", "continuous")
    if kind not in TYPES:
        raise ProblemError(f"variable {name}: type must be one of {TYPES}.")
    out = Variable(name=name, type=kind, unit=str(v.get("unit", "")), description=str(v.get("description", "")))
    if kind in ("continuous", "integer"):
        try:
            out.low, out.high = float(v["low"]), float(v["high"])
        except (KeyError, TypeError, ValueError):
            raise ProblemError(f"variable {name}: {kind} needs numeric low and high.")
        if not (math.isfinite(out.low) and math.isfinite(out.high) and out.low < out.high):
            raise ProblemError(f"variable {name}: needs finite low < high.")
        out.log = bool(v.get("log", False))
        if out.log and out.low <= 0:
            raise ProblemError(f"variable {name}: log scale needs low > 0.")
        if kind == "integer":
            if out.low != int(out.low) or out.high != int(out.high):
                raise ProblemError(f"variable {name}: integer bounds must be whole numbers.")
        elif v.get("resolution") is not None:
            out.resolution = float(v["resolution"])
            if not out.resolution > 0:
                raise ProblemError(f"variable {name}: resolution must be positive.")
    else:
        choices = v.get("choices")
        if not isinstance(choices, list) or len(choices) < 2 or len({str(c) for c in choices}) != len(choices):
            raise ProblemError(f"variable {name}: {kind} needs a list of at least two distinct choices.")
        if any("," in str(c) or "\n" in str(c) for c in choices):
            raise ProblemError(f"variable {name}: choices may not contain commas or newlines.")
        out.choices = choices
    return out
