"""One campaign: the skill's core, with the runner playing the user.

Each call, the agent follows the skill's METHOD.md and writes a proposal. The
runner commits it with the core (`aoptcore`) and then works through the block
exactly as a user would: the core's driver asks the block's optimizer (in a
sandboxed host) for a point, the runner evaluates the task there, records the
value in data.csv and passes it back, until the block is done.

Run directory:

    workspace/                  sandbox root and the agent's working directory
      skill/                    copy of the skill folder (METHOD.md, aopt, lib/, host.py, ...)
      .agentoptimizer/          the core's project folder
        problem.yaml PROBLEM.md space.json data.csv decisions.md   written by the runner
        [candidates.csv prior.csv]
        agent/ work/            agent-writable
        blocks/                 written by the runner (commit)
        state/                  host-writable (optimizer state)
      .venv .tmp .home
    private/                    journal.csv, policy, manifest, per-call and per-block logs
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from aoptcore import driver
from aoptcore import project as core

from . import data, sandbox

LOOP = Path(__file__).resolve().parent
REPO = LOOP.parents[3]
SKILL = REPO / "skill" / "agentoptimizer"
SKILL_IGNORE = shutil.ignore_patterns(".venv", "uv.lock", "tests", "install.sh", "__pycache__", ".DS_Store",
                                      "*.egg-info", ".pytest_cache")
MAX_FAILURES = 3  # consecutive calls without a valid proposal before the runner stops


@dataclass
class Campaign:
    task: object
    budget: int
    seed: int
    root: Path
    prefix: dict = field(default_factory=lambda: {"agent": [], "host": []})

    def __post_init__(self):
        self.workspace, self.private = self.root / "workspace", self.root / "private"
        self.folder = self.workspace / ".agentoptimizer"
        self.journal = data.Journal(self.task, self.private / "journal.csv")
        self.log: list[dict] = []  # one entry per call, saved to private/blocks.json
        self.written = self.decisions.read_text() if self.decisions.exists() else ""

    @property
    def decisions(self) -> Path:
        return self.folder / "decisions.md"

    def project(self) -> core.Project:
        return core.Project(self.folder)

    # ---- setup ---------------------------------------------------------------------------
    @classmethod
    def prepare(cls, task, budget: int, seed: int, root: Path) -> "Campaign":
        workspace, private = root / "workspace", root / "private"
        folder = workspace / ".agentoptimizer"
        for d in [".tmp", ".home"]:
            (workspace / d).mkdir(parents=True, exist_ok=True)
        private.mkdir(mode=0o700)
        shutil.copytree(SKILL, workspace / "skill", ignore=SKILL_IGNORE)
        folder.mkdir()
        (folder / "problem.yaml").write_text(yaml.safe_dump(problem_yaml(task, budget, seed, folder), sort_keys=False))
        (folder / "PROBLEM.md").write_text(problem_md(task, budget, seed))
        if getattr(task, "prior_path", None):
            shutil.copyfile(task.prior_path, folder / "prior.csv")
        core.init(folder)
        campaign = cls(task, budget, seed, root)
        campaign.written = campaign.decisions.read_text()
        return campaign

    # ---- loop ----------------------------------------------------------------------------
    def run(self, call_agent, max_calls: int | None = None) -> str:
        """Loop until the budget is spent or calls keep failing.
        `call_agent(number, prompt)` runs one skill call and returns its exit code."""
        failures = 0
        for number in range(1, (max_calls or self.budget + 10) + 1):
            if self.remaining == 0:
                return self.note("Budget exhausted. Campaign finished.", "budget")
            for name in ("proposal.json", "decision.md"):
                (self.folder / "agent" / name).unlink(missing_ok=True)
            code = call_agent(number, self.prompt(number))
            self.snapshot_context(number)
            try:
                if code != 0:
                    raise core.Rejected(f"the agent exited with code {code}.")
                p = self.project()
                proposal, decision = p.read_agent_files()
                record = p.commit(proposal, decision)
            except core.Rejected as e:
                failures += 1
                self.log.append({"call": number, "rejected": str(e)})
                self.save_log()
                self.note(f"Call {number} produced no valid proposal: {e} No evaluations were made.")
                if failures >= MAX_FAILURES:
                    return self.note(f"{MAX_FAILURES} consecutive calls failed. Campaign stopped.", "failures")
                continue
            self.written = self.decisions.read_text()
            failures = 0
            self.log.append({"call": number, "block": record["block"], "after": record["after"],
                             "proposal": {k: record[k] for k in core.FIELDS}})
            self.save_log()
            self.run_block(record)
        return self.note("Call limit reached. Campaign stopped.", "calls")

    @property
    def remaining(self) -> int:
        return self.budget - len(self.journal)

    def prompt(self, number: int) -> str:
        return (f"Skill call {number}. {len(self.journal)} of {self.budget} evaluations used; "
                f"{self.remaining} remaining. The skill folder is `skill/` and `aopt` is `skill/aopt`; "
                "use `skill/aopt python` for every Python script. Follow METHOD.md: review "
                ".agentoptimizer/data.csv, decisions.md and agent/context.md, then write agent/proposal.json "
                "and agent/decision.md, update agent/context.md, and run `skill/aopt validate`. The runner "
                "commits the block and evaluates it.")

    def run_block(self, record: dict) -> None:
        p = self.project()
        sandboxed = bool(self.prefix["host"])
        logs = self.private / "blocks"
        logs.mkdir(exist_ok=True)
        log = logs / f"{record['block']:03d}.log"

        def handle(script, space, space_path, state, args, mode):
            return driver.HostProcess(
                script, space_path, state, args, mode, prefix=self.prefix["host"],
                python=str(self.workspace / ".venv/bin/python") if sandboxed else driver.sys.executable,
                env=sandbox.environment(self.workspace) if sandboxed else None, log=log,
                host=self.workspace / "skill" / "host.py")

        session = driver.Session(p, record, handle)
        used = 0
        try:
            session.open()
            while True:
                rows = session.propose()
                if not rows:
                    break
                for r in rows:
                    ev, index, x = self.evaluate(p, r)
                    p.table.set_result(r["id"], ev.y, ev.g)
                    p.table.save()
                    self.journal.append(r, record["block"], record["label"], x, ev, index)
                    used += 1
                session.sync()
            session.close()
        except driver.HandleError as e:
            session.fail(str(e).replace(str(self.workspace) + "/", ""))
        self.written = self.decisions.read_text()
        for m in session.messages:
            self.note(f"Block {record['block']} ({record['label']}) ended early after {used} of "
                      f"{record['n_points']} evaluations: {m}")
        self.log[-1].update(used=used, finished=session.st.get("finished", False),
                            error=not session.st.get("saved", False))
        self.save_log()

    def evaluate(self, p: core.Project, r: dict):
        if p.problem.candidates is not None:
            index = int(r["candidate"])
            return self.task.evaluate_index(index), index, np.asarray(self.task.candidates[index], dtype=float)
        x = np.array([public_value(spec, r[f"x{d}"]) for d, spec in enumerate(variable_specs(self.task))])
        return self.task.evaluate(x), None, x

    # ---- records -------------------------------------------------------------------------
    def note(self, text: str, reason: str | None = None):
        self.project().note(text)
        self.written = self.decisions.read_text()
        return reason

    def save_log(self) -> None:
        (self.private / "blocks.json").write_text(json.dumps(self.log, indent=1))

    def snapshot_context(self, number: int) -> None:
        logs = self.private / "calls" / f"{number:03d}"
        logs.mkdir(parents=True, exist_ok=True)
        for name in ("context.md", "proposal.json", "decision.md"):
            src = self.folder / "agent" / name
            if src.is_file() and not src.is_symlink():
                shutil.copyfile(src, logs / name)


def problem_yaml(task, budget: int, seed: int, folder: Path) -> dict:
    """The task as the core's problem.yaml: anonymous variables x0.., objective y, minimized."""
    indexed = getattr(task, "indexed_pool", False)
    bounds = np.asarray(task.bounds, dtype=float).copy()
    flat = bounds[:, 0] >= bounds[:, 1]  # a constant pool feature: the core needs low < high
    bounds[flat] += [-0.5, 0.5]
    out = {"name": "benchmark",
           "objective": {"name": "y", "direction": "minimize"},
           "variables": [{"name": f"x{d}", **spec} for d, spec in enumerate(variable_specs(task, bounds))],
           "constraints": [{"name": f"g{i}"} for i in range(task.n_constraints)],
           # A nearest-mapping pool is not repeatable: allow repeats to be evaluated again.
           "noise": {"present": bool(task.noisy), "replicates_allowed": bool(task.noisy or (task.pool and not indexed))},
           "budget": budget, "seed": seed}
    if indexed:
        with open(folder / "candidates.csv", "w") as f:
            f.write(",".join(f"x{d}" for d in range(task.dim)) + "\n")
            for c in np.asarray(task.candidates, dtype=float):
                f.write(",".join(repr(float(v)) for v in c) + "\n")
        out["candidates"] = "candidates.csv"
    if not out["constraints"]:
        del out["constraints"]
    return out


def variable_specs(task, bounds=None) -> list[dict]:
    """problem.yaml entries (without names) for x0..: the task's own, or continuous boxes."""
    if task.variables is not None:
        return [dict(v) for v in task.variables]
    bounds = np.asarray(task.bounds, dtype=float) if bounds is None else bounds
    return [{"type": "continuous", "low": float(lo), "high": float(hi)} for lo, hi in bounds]


def public_value(spec: dict, value) -> float:
    """A data.csv value -> the task's numeric coordinate (a categorical's choice position)."""
    if spec["type"] == "categorical":
        return float(spec["choices"].index(str(value)))
    return float(value)


def problem_md(task, budget: int, seed: int) -> str:
    """The public problem statement. No domain knowledge: the identity is withheld."""
    indexed = getattr(task, "indexed_pool", False)
    kind = "noisy" if task.noisy else "deterministic"
    lines = ["# Problem\n",
             f"Minimize a {kind} black-box objective `y` of {task.dim} variables `x0`..`x{task.dim - 1}`."
             if task.variables is not None else
             f"Minimize a {kind} black-box objective `y` of {task.dim} real variables `x0`..`x{task.dim - 1}`.",
             "Each variable's type and range are in problem.yaml." if task.variables is not None else
             "Bounds per variable are in problem.yaml.",
             f"Budget: {budget} evaluations. Seed: {seed} (use it for optimizer randomness).",
             "The objective's identity is withheld: its implementation and optimum are private, and there is "
             "no domain knowledge. Reason from the data alone.",
             "Every point is evaluated by the runner as soon as it is requested."]
    if task.noisy:
        lines.append("Each evaluation returns a noisy observation; the campaign is scored by the noise-free value "
                     "at the point with the lowest observed y.")
    if task.n_constraints:
        lines.append(f"There are {task.n_constraints} black-box inequality constraints, columns g0.. in data.csv: "
                     "a point is feasible iff every g <= 0 (tolerance 1e-10). Infeasible points are still charged. "
                     "The result is the best feasible observation.")
    if task.pool and not indexed:
        lines.append("The objective is not repeatable: the value at x depends on which points were already "
                     "evaluated, so evaluating the same x again generally gives a different value.")
    if indexed:
        lines.append(f"This is a finite pool of {task.n_candidates} candidates listed in candidates.csv. "
                     "Optimizers propose candidate indices, not points; each candidate may be evaluated once.")
    if getattr(task, "prior_path", None):
        n = sum(1 for _ in open(task.prior_path)) - 1
        lines.append(f"prior.csv holds {n} earlier measurements of the same objective at other points in the same "
                     "feature space (columns x0.., y; none of them is a pool candidate's row). They are free: "
                     "they do not count against the budget and are not in data.csv.")
    return "\n".join(lines) + "\n"


def manifest(campaign: Campaign, cfg: dict, runtime: dict) -> dict:
    git = lambda *a: subprocess.check_output(["git", "-C", str(REPO), *a], text=True).strip()
    return {"task": cfg["task"], "budget": campaign.budget, "seed": campaign.seed, "agent": cfg["agent"],
            "model": cfg["model"], "effort": cfg["effort"], "context": cfg.get("context", "fresh"),
            "source_commit": git("rev-parse", "HEAD"), "source_status": git("status", "--short"),
            "agent_runtime": runtime}
