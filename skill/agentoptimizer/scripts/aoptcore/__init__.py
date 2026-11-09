"""The agentoptimizer skill's core: problem definition, encoding, data, blocks and the optimizer driver.

For analysis scripts:

    from aoptcore import observations
    X, y, rows = observations()   # encoded points, minimized objective, the data.csv rows
"""
from __future__ import annotations

from pathlib import Path


def observations(project=".agentoptimizer", include_failed: bool = False):
    """Measured rows as the optimizers see them: (X encoded, y minimized, rows).
    Pool problems give candidate indices in X. Failed rows are left out unless asked for."""
    import numpy as np

    from .project import Project
    p = Project(Path(project))
    statuses = ("measured", "failed") if include_failed else ("measured",)
    rows = p.table.by_status(*statuses)
    X = [p.encoder.encode(p.table.config(r)) for r in rows]
    y = [p.table.value(r) if r["status"] == "measured" else float("nan") for r in rows]
    return np.array(X, dtype=float).reshape(len(rows), p.encoder.dim), np.array(y, dtype=float), rows
