"""One campaign: call the skill, validate its proposal, run the block, repeat.

Run directory:

    workspace/            sandbox root; readable by agent and host
      SKILL.md PROBLEM.md space.json host.py check.py    fixed
      data.csv decisions.md                     written by the runner only
      lib/                library optimizers
      blocks/NNN_label.py snapshot of each block's optimizer (what the host loads)
      state/label.state   optimizer state, one per label (host-writable)
      agent/              proposal.json, decision.md, context.md (agent-writable)
      work/               agent scratch and custom optimizers (agent-writable)
    private/              journal.csv, policy, manifest, per-call logs
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import block, data, sandbox

LOOP = Path(__file__).resolve().parent
REPO = LOOP.parents[2]
LABEL = re.compile(r"[a-z][a-z0-9_]{0,39}")
MAX_FAILURES = 3  # consecutive calls without a valid proposal before the runner stops


class Rejected(Exception):
    pass


@dataclass
class Campaign:
    task: object
    budget: int
    seed: int
    root: Path
    prefix: dict = field(default_factory=lambda: {"agent": [], "host": []})

    def __post_init__(self):
        self.workspace, self.private = self.root / "workspace", self.root / "private"
        self.store = data.Store(self.task, self.workspace / "data.csv", self.private / "journal.csv")
        self.decisions = self.workspace / "decisions.md"
        self.blocks = 0
        self.labels: dict[str, dict] = {}  # label -> {"source": script text, "saved": bool}
        self.log: list[dict] = []  # one entry per accepted proposal, saved to private/blocks.json
        self.written = self.decisions.read_text() if self.decisions.exists() else ""

    # ---- setup ---------------------------------------------------------------------------

    @classmethod
    def prepare(cls, task, budget: int, seed: int, root: Path) -> "Campaign":
        workspace, private = root / "workspace", root / "private"
        for d in ["lib", "blocks", "state", "agent", "work", ".tmp", ".home"]:
            (workspace / d).mkdir(parents=True, exist_ok=True)
        private.mkdir(mode=0o700)
        shutil.copyfile(LOOP / "SKILL.md", workspace / "SKILL.md")
        for name in ("host.py", "check.py"):
            shutil.copyfile(LOOP / name, workspace / name)
        if (LOOP / "lib").is_dir():
            shutil.copytree(LOOP / "lib", workspace / "lib", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        (workspace / "PROBLEM.md").write_text(problem(task, budget, seed))
        data.write_space(task, workspace / "space.json")
        (workspace / "decisions.md").write_text("# Decisions\n")
        (workspace / "agent/context.md").write_text("")
        return cls(task, budget, seed, root)

    # ---- loop ----------------------------------------------------------------------------

    def run(self, call_agent, max_calls: int | None = None) -> str:
        """Loop until the budget is spent or calls keep failing.
        `call_agent(number, prompt)` runs one skill call and returns its exit code."""
        failures = 0
        for number in range(1, (max_calls or self.budget + 10) + 1):
            if self.remaining == 0:
                return self.note("Budget exhausted. Campaign finished.", "budget")
            for name in ("proposal.json", "decision.md"):
                (self.workspace / "agent" / name).unlink(missing_ok=True)
            code = call_agent(number, self.prompt(number))
            self.snapshot_context(number)
            try:
                proposal, decision = self.read_proposal(code)
            except Rejected as e:
                failures += 1
                self.note(f"Call {number} produced no valid proposal: {e} No evaluations were made.")
                if failures >= MAX_FAILURES:
                    return self.note(f"{MAX_FAILURES} consecutive calls failed. Campaign stopped.", "failures")
                continue
            failures = 0
            self.record(number, proposal, decision)
            self.run_block(proposal)
        return self.note("Call limit reached. Campaign stopped.", "calls")

    @property
    def remaining(self) -> int:
        return self.budget - len(self.store)

    def prompt(self, number: int) -> str:
        return (f"Skill call {number}. {len(self.store)} of {self.budget} evaluations used; "
                f"{self.remaining} remaining. Follow SKILL.md: review data.csv, decisions.md and "
                "agent/context.md, then write agent/proposal.json and agent/decision.md and update "
                "agent/context.md.")

    def read_proposal(self, code: int):
        agent = self.workspace / "agent"
        if code != 0:
            raise Rejected(f"the agent exited with code {code}.")
        for name in ("proposal.json", "decision.md"):
            if (agent / name).is_symlink():
                raise Rejected(f"agent/{name} is a symlink.")
        try:
            proposal = json.loads((agent / "proposal.json").read_text())
            decision = re.sub(r"\A## Decision after evaluation[^\n]*\n", "",
                              (agent / "decision.md").read_text().strip()).strip()  # the runner adds the heading
        except FileNotFoundError as e:
            raise Rejected(f"agent/{Path(e.filename).name} is missing.")
        except json.JSONDecodeError as e:
            raise Rejected(f"agent/proposal.json is not valid JSON ({e}).")
        if not decision:
            raise Rejected("agent/decision.md is empty.")
        return self.validate(proposal), decision

    def validate(self, p) -> dict:
        if not isinstance(p, dict):
            raise Rejected("the proposal is not a JSON object.")
        n = p.get("n_points")
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= self.remaining:
            raise Rejected(f"n_points must be an integer from 1 to {self.remaining}; "
                           "the campaign runs to the full budget.")
        unknown = set(p) - {"script", "label", "args", "n_points", "state", "warm_start"}
        if unknown:
            raise Rejected(f"unknown fields {sorted(unknown)}.")
        label, state = p.get("label"), p.get("state")
        if not isinstance(label, str) or not LABEL.fullmatch(label):
            raise Rejected("label must match [a-z][a-z0-9_]{0,39}.")
        if state not in ("continue", "restart"):
            raise Rejected('state must be "continue" or "restart".')
        if not isinstance(p.get("args", {}), dict) or not isinstance(p.get("warm_start", False), bool):
            raise Rejected("args must be an object and warm_start a boolean.")
        script = self.script_path(p.get("script"))
        source = script.read_text()
        if state == "continue":
            previous = self.labels.get(label)
            if previous is None or not previous["saved"]:
                raise Rejected(f'there is no saved state for label "{label}"; use "restart".')
            if previous["source"] != source:
                raise Rejected(f'{p["script"]} changed since "{label}" last ran; use "restart" or a new label.')
        return {"script": p["script"], "label": label, "args": p.get("args", {}), "n_points": n,
                "state": state, "warm_start": p.get("warm_start", False), "_source": source}

    def script_path(self, relative) -> Path:
        if not isinstance(relative, str):
            raise Rejected("script must be a path such as lib/cma_es.py or work/my_optimizer.py.")
        path = self.workspace / relative
        try:
            inside = path.resolve().relative_to(self.workspace.resolve())
        except ValueError:
            inside = None
        if inside is None or inside.parts[0] not in ("lib", "work") or path.suffix != ".py":
            raise Rejected("script must be a .py file under lib/ or work/.")
        if path.is_symlink() or not path.is_file():
            raise Rejected(f"{relative} does not exist or is not a regular file.")
        return path

    def record(self, number: int, proposal: dict, decision: str) -> None:
        shown = {k: v for k, v in proposal.items() if not k.startswith("_")}
        self.log.append({"call": number, "after": len(self.store), "proposal": shown})
        self.save_log()
        self.append(f"## Decision after evaluation {len(self.store)} (call {number})\n\n{decision}\n\n"
                    f"Proposal: `{json.dumps(shown)}`\n")

    def run_block(self, p: dict) -> None:
        self.blocks += 1
        script = self.workspace / "blocks" / f"{self.blocks:03d}_{p['label']}.py"
        script.write_text(p["_source"])
        b = block.Block(self.blocks, p["label"], script, p["args"], p["n_points"], p["state"], p["warm_start"])
        sandboxed = bool(self.prefix["host"])
        logs = self.private / "blocks"
        logs.mkdir(exist_ok=True)
        result = block.run(self.task, self.store, b, self.workspace / "state" / f"{p['label']}.state",
                           self.workspace / "space.json", logs / f"{self.blocks:03d}.log", self.prefix["host"],
                           str(self.workspace / ".venv/bin/python") if sandboxed else sys.executable,
                           sandbox.environment(self.workspace) if sandboxed else None,
                           host_script=self.workspace / "host.py")
        self.labels[p["label"]] = {"source": p["_source"], "saved": result.state_saved}
        self.log[-1].update(block=self.blocks, script=script.name, used=result.used,
                            finished=result.finished, error=result.error is not None)
        self.save_log()
        if result.finished:
            self.note(f"Block {self.blocks} ({p['label']}) ended early after {result.used} of "
                      f"{p['n_points']} evaluations: the optimizer reported it had finished "
                      "(converged or exhausted). Its state was saved.")
        if result.error:
            error = result.error.replace(str(self.workspace) + "/", "").strip()[-3000:]
            self.note(f"Block {self.blocks} ({p['label']}) stopped after {result.used} of {p['n_points']} "
                      f"evaluations. Its state was not saved, so the next block for this label must "
                      f"restart.\n\n```\n{error}\n```")

    # ---- records -------------------------------------------------------------------------

    def note(self, text: str, reason: str | None = None):
        self.append(f"## Runner note after evaluation {len(self.store)}\n\n{text}\n")
        return reason

    def append(self, text: str) -> None:
        self.written += "\n" + text
        with open(self.decisions, "a") as f:
            f.write("\n" + text)

    def save_log(self) -> None:
        (self.private / "blocks.json").write_text(json.dumps(self.log, indent=1))

    def snapshot_context(self, number: int) -> None:
        logs = self.private / "calls" / f"{number:03d}"
        logs.mkdir(parents=True, exist_ok=True)
        for name in ("context.md", "proposal.json", "decision.md"):
            src = self.workspace / "agent" / name
            if src.is_file() and not src.is_symlink():
                shutil.copyfile(src, logs / name)


def problem(task, budget: int, seed: int) -> str:
    """The public problem statement."""
    indexed = getattr(task, "indexed_pool", False)
    kind = "noisy" if task.noisy else "deterministic"
    lines = [f"# Problem\n",
             f"Minimize a {kind} black-box objective of {task.dim} real variables.",
             "Bounds per variable are in space.json.",
             f"Budget: {budget} evaluations. Seed: {seed} (use it for optimizer randomness).",
             "The objective's identity, implementation and optimum are private."]
    if task.noisy:
        lines.append("Each evaluation returns a noisy observation; the campaign is scored by the "
                     "noise-free value at the point with the lowest observed y.")
    if task.n_constraints:
        lines.append(f"There are {task.n_constraints} black-box inequality constraints, columns g0.. "
                     "in data.csv: a point is feasible iff every g <= 0 (tolerance 1e-10). Infeasible "
                     "points are still charged. The result is the best feasible observation.")
    if task.pool and not indexed:
        lines.append("The objective is not repeatable: the value at x depends on which points were "
                     "already evaluated, so evaluating the same x again generally gives a different value.")
    if indexed:
        lines.append(f"This is a finite pool of {task.n_candidates} candidates whose features are "
                     "space.json `candidates`. Optimizers propose candidate indices, not points; each "
                     "index may be evaluated once.")
    return "\n".join(lines) + "\n"


def manifest(campaign: Campaign, cfg: dict, runtime: dict) -> dict:
    git = lambda *a: subprocess.check_output(["git", "-C", str(REPO), *a], text=True).strip()
    return {"task": cfg["task"], "budget": campaign.budget, "seed": campaign.seed, "agent": cfg["agent"],
            "model": cfg["model"], "effort": cfg["effort"], "source_commit": git("rev-parse", "HEAD"),
            "source_status": git("status", "--short"), "agent_runtime": runtime}
