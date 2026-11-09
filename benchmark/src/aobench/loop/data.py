"""The private journal: every evaluation with what only scoring may see.

The public record is the core's data.csv (in the workspace). journal.csv adds
the noise-free value and timing, one row per evaluation, flushed and fsynced
before the next evaluation so a crash never loses a paid evaluation.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path


class Journal:
    def __init__(self, task, path: Path):
        self.path = Path(path)
        self.indexed = getattr(task, "indexed_pool", False)
        self.dim, self.n_g = task.dim, task.n_constraints
        self.columns = (["eval", "id", "block", "optimizer"] + (["candidate_index"] if self.indexed else [])
                        + [f"x{d}" for d in range(self.dim)] + ["y"] + [f"g{i}" for i in range(self.n_g)]
                        + (["feasible"] if self.n_g else []) + ["y_true", "eval_seconds"])
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.columns)
        with open(self.path, newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def append(self, row: dict, block: int, label: str, x, ev, index=None) -> None:
        out = {"eval": len(self.rows) + 1, "id": row["id"], "block": block, "optimizer": label,
               "candidate_index": index, "y": ev.y, "feasible": ev.feasible, "y_true": ev.y_true,
               "eval_seconds": ev.elapsed}
        out.update({f"x{d}": float(v) for d, v in enumerate(x)})
        out.update({f"g{i}": v for i, v in enumerate(ev.g)})
        text = {c: "" if out.get(c) is None else str(out[c]) for c in self.columns}
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([text[c] for c in self.columns])
            f.flush()
            os.fsync(f.fileno())
        self.rows.append(text)
