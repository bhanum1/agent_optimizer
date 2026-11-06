"""Propose mode: the agent's next block for data measured outside the runner.

The data is a CSV with columns x0, x1, ..., y (minimized). It becomes the run's
data.csv, and `UserData` stands in for the task: it carries the space, while
its objective refuses to be called, since the block is run outside the runner.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from agentoptimizer.tasks.base import Evaluation, Task


class UserData(Task):
    def __init__(self, path: Path, bounds: str):
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            rows = list(reader)
        self.dim = len(fields) - 1
        if self.dim < 1 or fields != [f"x{i}" for i in range(self.dim)] + ["y"]:
            raise ValueError("Expected columns x0,x1,...,y.")
        if not rows:
            raise ValueError("The CSV has no rows.")
        self.X = np.array([[float(r[f"x{i}"]) for i in range(self.dim)] for r in rows])
        self.y = np.array([float(r["y"]) for r in rows])
        if not (np.isfinite(self.X).all() and np.isfinite(self.y).all()):
            raise ValueError("Every x and y must be a finite number.")
        self.bounds = parse_bounds(bounds, self.dim)
        if ((self.X < self.bounds[:, 0]) | (self.X > self.bounds[:, 1])).any():
            raise ValueError("Some rows lie outside --bounds.")
        self.name = f"user_data_d{self.dim}_n{len(rows)}"
        self.f_opt = float("nan")  # unknown; propose mode is never scored

    def _f(self, x):
        raise RuntimeError("Propose mode never evaluates the objective.")

    def fill(self, store) -> None:
        """Write every row to the run's data.csv as block 0, optimizer "user_data"."""
        for x, y in zip(self.X, self.y):
            store.append(0, "user_data", x, Evaluation(float(y), (), 0.0))


def parse_bounds(text: str, dim: int) -> np.ndarray:
    """`lo,hi` for every variable, or a JSON list of [lo, hi] pairs (inline or a file)."""
    if Path(text).is_file():
        text = Path(text).read_text()
    text = text.strip()
    pairs = json.loads(text) if text.startswith("[") else [[float(v) for v in text.split(",")]]
    bounds = np.array(pairs * dim if len(pairs) == 1 else pairs, dtype=float)
    if bounds.shape != (dim, 2) or not (bounds[:, 0] < bounds[:, 1]).all():
        raise ValueError(f"--bounds must give {dim} [low, high] pairs with low < high.")
    return bounds
