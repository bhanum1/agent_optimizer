"""The .agentoptimizer folder: problem, data, blocks, optimizer state, and the agent's notes.

    problem.yaml PROBLEM.md      the definition (machine-readable, and narrative)
    data.csv                     every requested and measured point
    space.json                   the encoded space optimizers are built with
    decisions.md                 one entry per block, append-only
    agent/                       proposal.json, decision.md, context.md (the agent's files)
    blocks/NNN_label.py/.json    each block's optimizer source and proposal
    state/label.state/.json      each label's optimizer state and bookkeeping
    work/                        the agent's scripts and custom optimizers
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import problem as problem_mod
from .data import Table
from .encoding import Encoder

SKILL = Path(__file__).resolve().parents[2]  # the skill folder: host.py, check.py, METHOD.md
LABEL = re.compile(r"[a-z][a-z0-9_]{0,39}")
FIELDS = {"script", "label", "args", "n_points", "state", "warm_start"}
DIRS = ["agent", "blocks", "state", "work"]


class Rejected(ValueError):
    """A proposal that cannot be committed; the message says why."""


class Project:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not (self.root / "problem.yaml").is_file():
            raise FileNotFoundError(f"{self.root / 'problem.yaml'} not found; run `aopt init` first.")
        self.problem = problem_mod.load(self.root / "problem.yaml")
        self.encoder = Encoder(self.problem)
        self.table = Table(self.root / "data.csv", self.problem, self.encoder)

    # ---- paths ----------------------------------------------------------------------------------
    @property
    def space_path(self) -> Path:
        return self.root / "space.json"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions.md"

    def state_path(self, label: str) -> Path:
        return self.root / "state" / f"{label}.state"

    # ---- budget -----------------------------------------------------------------------------------
    @property
    def remaining(self) -> int:
        return self.problem.budget - self.table.used

    # ---- blocks -----------------------------------------------------------------------------------
    def blocks(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted((self.root / "blocks").glob("[0-9][0-9][0-9]_*.json"))]

    def current(self) -> dict | None:
        blocks = self.blocks()
        return blocks[-1] if blocks else None

    def label_state(self, label: str) -> dict:
        path = self.root / "state" / f"{label}.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_label_state(self, label: str, st: dict) -> None:
        (self.root / "state" / f"{label}.json").write_text(json.dumps(st, indent=1))

    def block_progress(self, block: dict) -> dict:
        rows = self.table.block_rows(block["block"])
        st = self.label_state(block["label"])
        finished = st.get("block") == block["block"] and st.get("finished", False)
        pending = [r for r in rows if r["status"] == "pending"]
        left = 0 if finished else min(block["n_points"] - len(rows), self.remaining)
        return {"rows": rows, "pending": pending, "left": max(left, 0), "finished": finished,
                "complete": not pending and (finished or left <= 0)}

    # ---- proposals ----------------------------------------------------------------------------------
    def script_path(self, relative) -> Path:
        if not isinstance(relative, str) or not relative.endswith(".py"):
            raise Rejected("script must be a .py path such as work/my_optimizer.py.")
        parts = Path(relative).parts
        if len(parts) < 2 or parts[0] != "work" or ".." in parts:
            raise Rejected("script must be a .py file under work/.")
        path = self.root / relative
        if path.is_symlink() or not path.is_file():
            raise Rejected(f"{relative} does not exist or is not a regular file.")
        return path

    def validate(self, p, cancelling: int = 0) -> dict:
        """Check a proposal; return it normalized with its script source under `_source`.
        `cancelling` pending rows will be cancelled when it is committed (they free budget)."""
        if not isinstance(p, dict):
            raise Rejected("the proposal is not a JSON object.")
        unknown = set(p) - FIELDS
        if unknown:
            raise Rejected(f"unknown fields {sorted(unknown)}; allowed: {sorted(FIELDS)}.")
        remaining = self.remaining + cancelling
        n = p.get("n_points")
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= remaining:
            raise Rejected(f"n_points must be an integer from 1 to {remaining} (the remaining budget).")
        label, state = p.get("label"), p.get("state", "restart")
        if not isinstance(label, str) or not LABEL.fullmatch(label):
            raise Rejected("label must match [a-z][a-z0-9_]{0,39}.")
        if state not in ("continue", "restart"):
            raise Rejected('state must be "continue" or "restart".')
        args, warm = p.get("args", {}), p.get("warm_start", False)
        if not isinstance(args, dict) or not isinstance(warm, bool):
            raise Rejected("args must be an object and warm_start a boolean.")
        source = self.script_path(p.get("script")).read_text()
        if state == "continue":
            st = self.label_state(label)
            if not st.get("saved"):
                raise Rejected(f'there is no cleanly saved state for label "{label}"; use "restart".')
            if (self.root / st["source_file"]).read_text() != source:
                raise Rejected(f'{p["script"]} changed since "{label}" last ran; use "restart" or a new label.')
        return {"script": p["script"], "label": label, "args": args, "n_points": n, "state": state,
                "warm_start": warm, "_source": source}

    def read_agent_files(self) -> tuple[dict, str]:
        agent = self.root / "agent"
        for name in ("proposal.json", "decision.md"):
            if (agent / name).is_symlink():
                raise Rejected(f"agent/{name} is a symlink.")
        try:
            proposal = json.loads((agent / "proposal.json").read_text())
            decision = re.sub(r"\A## Block \d+[^\n]*\n", "", (agent / "decision.md").read_text().strip()).strip()
        except FileNotFoundError as e:
            raise Rejected(f"agent/{Path(e.filename).name} is missing.")
        except json.JSONDecodeError as e:
            raise Rejected(f"agent/proposal.json is not valid JSON ({e}).")
        if not decision:
            raise Rejected("agent/decision.md is empty.")
        return proposal, decision

    def commit(self, proposal: dict, decision: str, settle=None) -> dict:
        """Validate and record a new block. Earlier pending rows are cancelled; `settle(project)`
        first tells the previous block's optimizer its last results, so `continue` resumes exactly."""
        stale = self.table.by_status("pending")
        p = self.validate(proposal, cancelling=len(stale))
        if settle is not None:
            settle(self)
        for r in stale:
            r["status"] = "cancelled"
        number = len(self.blocks()) + 1
        stem = f"{number:03d}_{p['label']}"
        (self.root / "blocks" / f"{stem}.py").write_text(p["_source"])
        record = {k: v for k, v in p.items() if not k.startswith("_")}
        record.update(block=number, source_file=f"blocks/{stem}.py", after=self.table.used,
                      committed=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        (self.root / "blocks" / f"{stem}.json").write_text(json.dumps(record, indent=1))
        self.table.save()
        shown = {k: record[k] for k in ("script", "label", "args", "n_points", "state", "warm_start")}
        note = (f"\n\n{len(stale)} pending point(s) of earlier blocks were cancelled." if stale else "")
        self.append(f"## Block {number} (after {record['after']} of {self.problem.budget} measurements)\n\n"
                    f"{decision}\n\nProposal: `{json.dumps(shown)}`{note}\n")
        return record

    def append(self, text: str) -> None:
        with open(self.decisions, "a") as f:
            f.write("\n" + text)

    def note(self, text: str) -> None:
        self.append(f"## Note after {self.table.used} measurements\n\n{text}\n")


def init(root: Path, import_csv: Path | None = None) -> Project:
    """Create the folder around an existing problem.yaml; optionally import initial data."""
    root = Path(root)
    for d in DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)
    project = Project(root)
    if project.table.rows and import_csv:
        raise ValueError("data.csv already has rows; add new data to it directly.")
    if import_csv:
        from .data import import_rows
        import_rows(project.table, import_csv)
    project.table.save()
    project.space_path.write_text(json.dumps(project.encoder.space()))
    if not project.decisions.exists():
        project.decisions.write_text("# Decisions\n")
    for name in ("context.md",):
        (root / "agent" / name).touch()
    if not (root / "PROBLEM.md").exists():
        (root / "PROBLEM.md").write_text(f"# {project.problem.name}\n\n(Goal, domain knowledge and notes.)\n")
    return project
