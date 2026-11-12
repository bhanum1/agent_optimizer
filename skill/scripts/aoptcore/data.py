"""data.csv: every requested and measured point, in the user's own units.

    id,block,label,status,replicate_of,[candidate,]<variables...>,<objective>,<constraints...>,note

status:
    pending     requested; the objective cell is blank until measured
    measured    the objective (and any constraints) are filled in
    failed      the experiment could not be run or gave no value
    cancelled   requested but abandoned when a new block started; not counted

A row whose objective cell is filled counts as measured whatever its status says,
so filling in the number is enough. Rows the user adds without an id or block are
external data: they get an id and block 0. Extra columns the user adds are kept.
"""
from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path

from .encoding import Encoder, format_value, parse_value
from .problem import Problem

FIXED = ["id", "block", "label", "status", "replicate_of"]
STATUSES = ("pending", "measured", "failed", "cancelled")


class DataError(ValueError):
    pass


def header(problem: Problem) -> list[str]:
    return FIXED + (["candidate"] if problem.candidates is not None else []) + problem.columns + ["note"]


class Table:
    """data.csv in memory. Rows are dicts of strings; `parsed(row)` gives typed values."""

    def __init__(self, path: Path, problem: Problem, encoder: Encoder | None = None):
        self.path, self.problem = Path(path), problem
        self.encoder = encoder or Encoder(problem)
        self.columns = header(problem)
        self.rows: list[dict] = []
        if self.path.exists():
            self.read()

    # ---- reading ------------------------------------------------------------------------------
    def read(self) -> None:
        """(Re)load data.csv from disk."""
        self.rows = []
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = [dict(r) for r in reader]
        missing = [c for c in self.columns if c not in fields]
        if missing:
            raise DataError(f"data.csv is missing columns {missing}.")
        self.columns = self.columns + [c for c in fields if c not in self.columns]
        next_id = 1 + max([int(r["id"]) for r in rows if (r.get("id") or "").strip()], default=0)
        for n, r in enumerate(rows, start=2):
            r = {c: (r.get(c) or "").strip() for c in self.columns}
            if not any(r[c] for c in self.problem.columns):
                continue  # blank line
            if not r["id"]:
                r["id"], next_id = str(next_id), next_id + 1
                r["block"] = r["block"] or "0"
                r["label"] = r["label"] or "external"
            try:
                self._normalize(r)
            except (ValueError, KeyError) as e:
                raise DataError(f"data.csv line {n}: {e}")
            self.rows.append(r)
        ids = [r["id"] for r in self.rows]
        if len(set(ids)) != len(ids):
            raise DataError("data.csv has repeated ids.")

    def _normalize(self, r: dict) -> None:
        int(r["id"]), int(r["block"])
        for v in self.problem.variables:
            r[v.name] = format_value(v, parse_value(v, r[v.name]))
        if self.problem.candidates is not None:
            index = self.encoder.candidate_index({v.name: r[v.name] for v in self.problem.variables})
            if index is None:
                raise ValueError("the configuration is not one of the candidates")
            r["candidate"] = str(index)
        y = r[self.problem.objective]
        if y:
            value = float(y)
            if not math.isfinite(value):
                raise ValueError(f"{self.problem.objective} must be a finite number")
            for c in self.problem.constraints:
                if not r[c] or not math.isfinite(float(r[c])):
                    raise ValueError(f"measured rows need a number in constraint column {c}")
            r["status"] = "measured"
        else:
            r["status"] = r["status"] or "pending"
            if r["status"] == "measured" or (r["status"] == "pending" and r["block"] == "0"):
                raise ValueError(f"no value in {self.problem.objective}: fill it in, or set status to failed")
        if r["status"] not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")

    # ---- views --------------------------------------------------------------------------------
    def config(self, r: dict) -> dict:
        return {v.name: parse_value(v, r[v.name]) for v in self.problem.variables}

    def value(self, r: dict) -> float:
        """The objective as optimizers see it: minimized."""
        y = float(r[self.problem.objective])
        return -y if self.problem.maximize else y

    def g(self, r: dict) -> list[float]:
        return [float(r[c]) for c in self.problem.constraints]

    def feasible(self, r: dict) -> bool:
        return all(v <= 1e-10 for v in self.g(r))

    def by_status(self, *statuses) -> list[dict]:
        return [r for r in self.rows if r["status"] in statuses]

    @property
    def used(self) -> int:
        """Measurements charged to the budget: everything but cancelled rows."""
        return sum(r["status"] != "cancelled" for r in self.rows)

    def block_rows(self, block: int) -> list[dict]:
        return [r for r in self.rows if int(r["block"]) == block and r["status"] != "cancelled"]

    def best(self) -> dict | None:
        rows = [r for r in self.by_status("measured") if self.feasible(r)]
        return min(rows, key=self.value) if rows else None

    def penalty(self) -> float:
        """The value a failed row is told as: worse than every measured value."""
        values = [self.value(r) for r in self.by_status("measured")]
        if not values:
            return 1e6
        lo, hi = min(values), max(values)
        return hi + (hi - lo or abs(hi) or 1.0)

    def find(self, config: dict, statuses=("measured",)) -> dict | None:
        key = self.encoder.key(config)
        for r in self.rows:
            if r["status"] in statuses and self.encoder.key(self.config(r)) == key:
                return r
        return None

    def get(self, row_id) -> dict:
        for r in self.rows:
            if r["id"] == str(row_id):
                return r
        raise KeyError(f"no row with id {row_id}")

    # ---- writing ------------------------------------------------------------------------------
    def add(self, block: int, label: str, config: dict, replicate_of=None, note: str = "") -> dict:
        next_id = 1 + max([int(r["id"]) for r in self.rows], default=0)
        r = {c: "" for c in self.columns}
        r.update(id=str(next_id), block=str(block), label=label, status="pending",
                 replicate_of="" if replicate_of is None else str(replicate_of), note=note)
        for v in self.problem.variables:
            r[v.name] = format_value(v, config[v.name])
        if self.problem.candidates is not None:
            r["candidate"] = str(self.encoder.candidate_index(config))
        self.rows.append(r)
        return r

    def set_result(self, row_id, y: float, g=()) -> None:
        r = self.get(row_id)
        r[self.problem.objective] = repr(float(y))
        for c, v in zip(self.problem.constraints, g):
            r[c] = repr(float(v))
        r["status"] = "measured"

    def save(self) -> None:
        """Rewrite data.csv atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".data-", suffix=".csv")
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


def read_candidates(problem: Problem) -> list[dict]:
    with open(problem.candidates, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [v.name for v in problem.variables if v.name not in (reader.fieldnames or [])]
        if missing:
            raise DataError(f"{problem.candidates.name} is missing variable columns {missing}.")
        out = []
        for n, r in enumerate(reader, start=2):
            try:
                out.append({v.name: parse_value(v, r[v.name]) for v in problem.variables})
            except ValueError as e:
                raise DataError(f"{problem.candidates.name} line {n}: {e}")
    if not out:
        raise DataError(f"{problem.candidates.name} has no rows.")
    return out


def import_rows(table: Table, path: Path) -> int:
    """Append every row of a user CSV (variable and objective columns by name) as external data."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        need = [v.name for v in table.problem.variables] + [table.problem.objective] + table.problem.constraints
        missing = [c for c in need if c not in fields]
        if missing:
            raise DataError(f"{Path(path).name} is missing columns {missing}; it has {fields}.")
        rows = list(reader)
    added = 0
    for n, src in enumerate(rows, start=2):
        if not any((src.get(c) or "").strip() for c in need):
            continue
        r = {c: "" for c in table.columns}
        r.update({c: (src.get(c) or "").strip() for c in table.columns if c in src})
        r.update(id=str(1 + max([int(x["id"]) for x in table.rows], default=0)), block="0",
                 label="initial", replicate_of="")
        if not r["status"]:
            r["status"] = "measured" if r[table.problem.objective] else "failed"
        try:
            table._normalize(r)
        except (ValueError, KeyError) as e:
            raise DataError(f"{Path(path).name} line {n}: {e}")
        table.rows.append(r)
        added += 1
    return added
