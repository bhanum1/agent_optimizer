"""The campaign's evaluations: a public data.csv and a private journal.csv.

data.csv is what the agent and optimizers see. journal.csv adds what only
scoring may see (noise-free y, timing). Every row is flushed and fsynced
before the next evaluation, so a crash never loses a paid evaluation.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


def space(task) -> dict:
    """The public description of the search space handed to optimizers."""
    out = {"dim": task.dim, "bounds": np.asarray(task.bounds).tolist(),
           "n_constraints": task.n_constraints, "noisy": task.noisy,
           "indexed_pool": getattr(task, "indexed_pool", False), "candidates": None}
    if out["indexed_pool"]:  # a nearest-mapping pool keeps its candidates private
        out["candidates"] = np.asarray(task.candidates).tolist()
    return out


class Store:
    def __init__(self, task, public: Path, private: Path):
        self.indexed = getattr(task, "indexed_pool", False)
        self.public, self.private = Path(public), Path(private)
        self.columns = (["eval", "block", "optimizer"]
                        + (["candidate_index"] if self.indexed else [])
                        + [f"x{d}" for d in range(task.dim)] + ["y"]
                        + [f"g{i}" for i in range(task.n_constraints)]
                        + (["feasible"] if task.n_constraints else []))
        self.private_columns = self.columns + ["y_true", "eval_seconds"]
        for path, columns in ((self.public, self.columns), (self.private, self.private_columns)):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(columns)
        self.rows = self.read()

    def read(self) -> list[dict]:
        with open(self.private, newline="") as f:
            return list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def append(self, block: int, optimizer: str, x, ev, candidate_index=None) -> None:
        row = {"eval": len(self.rows) + 1, "block": block, "optimizer": optimizer,
               "candidate_index": candidate_index, "y": ev.y,
               "feasible": ev.feasible, "y_true": ev.y_true, "eval_seconds": ev.elapsed}
        row.update({f"x{d}": float(v) for d, v in enumerate(x)})
        row.update({f"g{i}": v for i, v in enumerate(ev.g)})
        text = {c: "" if row[c] is None else str(row[c]) for c in self.private_columns}
        for path, columns in ((self.public, self.columns), (self.private, self.private_columns)):
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow([text[c] for c in columns])
                f.flush()
                os.fsync(f.fileno())
        self.rows.append(text)

    def history(self):
        """(X, y, g) of every evaluation so far, in the form optimizers are told.
        X holds candidate indices for an indexed pool, points otherwise."""
        X, y, g = [], [], []
        dim = sum(c.startswith("x") for c in self.columns)
        n_g = sum(c.startswith("g") for c in self.columns)
        for r in self.rows:
            X.append(int(r["candidate_index"]) if self.indexed
                     else [float(r[f"x{d}"]) for d in range(dim)])
            y.append(float(r["y"]))
            g.append([float(r[f"g{i}"]) for i in range(n_g)])
        return X, y, g

    def used_indices(self) -> set[int]:
        return {int(r["candidate_index"]) for r in self.rows} if self.indexed else set()


def write_space(task, path: Path) -> None:
    path.write_text(json.dumps(space(task)))
