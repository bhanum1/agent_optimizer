"""Run the agent on the benchmark: the harness plays the user of the skill.

    python -m aobench.agent --out benchmark/runs/agent --tasks gallagher --seeds 1
    python -m aobench.agent --out benchmark/runs/try --tasks gallagher --agent fake

One campaign, for one task and seed:
1. Write the problem definition (`problem.yaml`, `PROBLEM.md`) into a fresh
   workspace that holds a copy of the skill, and run `aopt init`.
2. Call the agent: a fresh `claude -p` in the workspace, told to follow the
   skill's METHOD.md. It writes a proposal and its optimizer.
3. `aopt commit`, then `aopt suggest`; evaluate the task at every pending row of
   `data.csv`, write the values in, and `aopt suggest` again, until the block is
   done. This is exactly what a user does by hand.
4. Repeat from 2 until the budget is spent.

The harness talks to the skill only through `aopt` and the project files, as a
user would; it imports nothing from the skill. The agent sees only the
workspace: anonymous variables x0.., objective y, and no task identity.

Layout, per campaign:

    <work>/<task>/seed_<s>/     the agent's working directory (kept outside the repository)
      skill/                    copy of the skill folder
      .agentoptimizer/          the project: problem.yaml, data.csv, decisions.md, agent/, work/, ...
    <out>/<task>/agent/seed_<s>.csv           eval, y, regret: the same format as the baselines
    <out>/<task>/agent/seed_<s>/calls.jsonl   one line per agent call: blocks, cost, time
    <out>/<task>/agent/seed_<s>/calls/NNN/    prompt, event stream, stderr, the agent's outputs

Finished campaigns (their CSV exists) are skipped. An unfinished one is not
resumed: its workspace is removed and it starts again.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .run import FLOOR
from .tasks import SPECS, make_task

REPO = Path(__file__).resolve().parents[3]
SKILL = REPO / "skill" / "agentoptimizer"
SKILL_IGNORE = shutil.ignore_patterns(".venv", "uv.lock", "tests", "install.sh", "__pycache__", ".DS_Store",
                                      "*.egg-info", ".pytest_cache", ".python")
TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
# `aopt suggest` output that is not news: everything else it prints is a message about the block.
ROUTINE = ("Requested", "Still waiting", " ", "Measure them", "The budget is spent", "Block ")
MAX_FAILURES = 3  # consecutive calls that yield no evaluation before the campaign stops

PROMPT = """You are deciding the next block of an optimization campaign. The skill is in `skill/`: \
read `skill/METHOD.md` and follow it. The project folder is `.agentoptimizer/`, and `aopt` is `skill/aopt`, \
run from this directory. {used} of {budget} measurements are used; {remaining} remain. When `aopt validate` \
accepts your proposal, stop: the harness commits the block and measures its points, then calls you again.{note}"""


# ---- the problem definition --------------------------------------------------------------------
def problem_yaml(task) -> str:
    spec = task.spec
    lines = [f"name: bbob_{task.name}_s{task.seed}",
             "objective: {name: y, direction: minimize}",
             "variables:"]
    lines += [f"  - {{name: x{i}, type: continuous, low: -5, high: 5}}" for i in range(spec.dim)]
    noisy = spec.beta > 0
    lines += [f"noise: {{present: {str(noisy).lower()}, replicates_allowed: {str(noisy).lower()}}}",
              f"budget: {spec.budget}", "parallel: 1", f"seed: {task.seed}"]
    return "\n".join(lines) + "\n"


def problem_md(task) -> str:
    spec = task.spec
    noise = ("Repeated measurements of the same point differ: the objective is noisy, by an unknown amount. "
             "Replicates are allowed." if spec.beta > 0 else
             "The objective is deterministic: the same point always gives the same value.")
    return f"""# Problem

Minimize `y`, a black-box function of {spec.dim} continuous variables `x0`..`x{spec.dim - 1}`, each in [-5, 5].

The objective's identity is withheld: there is no domain knowledge about it.

{noise}

Each measurement is cheap and quick, and is made one at a time. The budget is {spec.budget} measurements, with \
no earlier data. The result is the lowest `y` measured.
"""


# ---- one campaign ------------------------------------------------------------------------------
class Campaign:
    def __init__(self, task_key: str, seed: int, out: Path, work: Path, agent: str, model: str | None,
                 effort: str | None, call_timeout: float, max_calls: int):
        self.task = make_task(task_key, seed=seed)
        self.budget = self.task.spec.budget
        self.result = out / task_key / "agent" / f"seed_{seed}.csv"
        self.logs = out / task_key / "agent" / f"seed_{seed}"
        self.workspace = work / task_key / f"seed_{seed}"
        self.project = self.workspace / ".agentoptimizer"
        self.agent, self.model, self.effort = agent, model, effort
        self.call_timeout, self.max_calls = call_timeout, max_calls
        self.rows: list[tuple] = []  # eval, y, regret: what the result CSV holds
        self.best_y, self.regret = np.inf, np.inf

    def prepare(self) -> None:
        for d in (self.workspace, self.logs):
            if d.exists():
                shutil.rmtree(d)
        self.logs.mkdir(parents=True)
        shutil.copytree(SKILL, self.workspace / "skill", ignore=SKILL_IGNORE)
        # aopt runs in the skill's own environment rather than building one per workspace.
        (self.workspace / "skill" / ".python").write_text(str(skill_python()))
        self.project.mkdir()
        (self.project / "problem.yaml").write_text(problem_yaml(self.task))
        (self.project / "PROBLEM.md").write_text(problem_md(self.task))
        self.aopt("init")

    def aopt(self, *args, check=True) -> subprocess.CompletedProcess:
        out = subprocess.run([str(self.workspace / "skill" / "aopt"), *args], cwd=self.workspace,
                             capture_output=True, text=True)
        if check and out.returncode != 0:
            raise RuntimeError(f"aopt {' '.join(args)} failed:\n{out.stdout}{out.stderr}")
        return out

    def run(self) -> str:
        self.prepare()
        failures, note = 0, ""
        for number in range(1, self.max_calls + 1):
            remaining = self.budget - len(self.rows)
            if remaining <= 0:
                return self.finish("budget")
            for name in ("proposal.json", "decision.md"):
                (self.project / "agent" / name).unlink(missing_ok=True)
            prompt = PROMPT.format(used=len(self.rows), budget=self.budget, remaining=remaining, note=note)
            t0 = time.time()
            info = self.call(number, prompt)
            commit = self.aopt("commit", check=False)
            entry = {"call": number, "seconds": round(time.time() - t0, 1), **info,
                     "committed": commit.returncode == 0}
            used = 0
            if commit.returncode == 0:
                proposal = json.loads((self.project / "agent" / "proposal.json").read_text())
                entry["proposal"] = proposal
                used, messages = self.run_block()
                entry["messages"] = messages
                note = ("" if not messages else
                        " The last block ended early: " + " ".join(messages) + " Details are in decisions.md.")
            else:
                error = (commit.stderr or commit.stdout).strip()
                entry["rejected"] = error
                note = f" Your previous call produced no block that could be committed: {error}"
            entry["evaluations"] = used
            with open(self.logs / "calls.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
            failures = 0 if used else failures + 1
            if failures >= MAX_FAILURES:
                return self.finish("failures")
        return self.finish("calls")

    def call(self, number: int, prompt: str) -> dict:
        """One agent call. Returns what its final event reports (cost, turns, exit code)."""
        logs = self.logs / "calls" / f"{number:03d}"
        logs.mkdir(parents=True)
        (logs / "prompt.txt").write_text(prompt)
        command, env = agent_command(self.agent, self.model, self.effort)
        with open(logs / "events.jsonl", "w") as events, open(logs / "stderr.log", "w") as errors:
            process = subprocess.Popen(command, cwd=self.workspace, env=env, stdin=subprocess.PIPE,
                                       stdout=events, stderr=errors, text=True)
            try:
                process.communicate(prompt, timeout=self.call_timeout)
                code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = "timeout"
        for name in ("proposal.json", "decision.md", "context.md"):
            src = self.project / "agent" / name
            if src.is_file():
                shutil.copyfile(src, logs / name)
        return {"exit": code, **final_event(logs / "events.jsonl")}

    def run_block(self) -> tuple[int, list[str]]:
        """Work through the committed block as a user would. Returns evaluations made and aopt's messages."""
        used, messages = 0, []
        while True:
            out = self.aopt("suggest", check=False)
            if out.returncode != 0:
                messages.append((out.stderr or out.stdout).strip())
                break
            messages += [line for line in out.stdout.splitlines() if line and not line.startswith(ROUTINE)]
            n = self.measure_pending()
            if n == 0:
                break
            used += n
        return used, messages

    def measure_pending(self) -> int:
        """Evaluate the task at every pending row of data.csv and write the values in."""
        path = self.project / "data.csv"
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        n = 0
        for r in rows:
            if r["status"] != "pending" or len(self.rows) >= self.budget:
                continue
            x = np.array([float(r[f"x{i}"]) for i in range(self.task.dim)])
            ev = self.task.evaluate(x)
            r["y"] = repr(ev.y)
            if ev.y < self.best_y:
                self.best_y, self.regret = ev.y, ev.y_true - self.task.f_opt
            self.rows.append((len(self.rows) + 1, ev.y, self.regret))
            n += 1
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(path)
        return n

    def finish(self, reason: str) -> str:
        (self.logs / "end.json").write_text(json.dumps({"reason": reason, "evaluations": len(self.rows),
                                                        "budget": self.budget}))
        if self.rows:
            # A campaign that stopped early keeps its best value for the rest of the budget.
            rows = self.rows + [(i, np.nan, self.regret) for i in range(len(self.rows) + 1, self.budget + 1)]
            tmp = self.result.with_suffix(".tmp")
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["eval", "y", "regret"])
                w.writerows(rows)
            tmp.replace(self.result)
        return reason


# ---- launching the agent -----------------------------------------------------------------------
def skill_python() -> Path:
    python = SKILL / ".venv" / "bin" / "python"
    if not python.exists():
        subprocess.run(["uv", "sync", "--project", str(SKILL)], check=True)
    return python


def agent_command(agent: str, model: str | None, effort: str | None):
    if agent == "fake":
        return [sys.executable, "-m", "aobench.fake_agent"], dict(os.environ)
    # A fresh, headless Claude Code: file and shell tools only, no web, no user settings,
    # memory, skills or MCP servers, and no session kept between calls.
    command = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--no-session-persistence",
               "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
               "--tools", ",".join(TOOLS), "--allowedTools", *TOOLS, "--permission-mode", "dontAsk"]
    if model:
        command += ["--model", model]
    if effort:
        command += ["--effort", effort]
    env = {**os.environ, "ENABLE_CLAUDEAI_MCP_SERVERS": "false", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    return command, env


def final_event(events: Path) -> dict:
    """Cost, turns and time from the last `result` event of a stream-json log, if any."""
    info = {}
    for line in events.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "result":
            info = {k: e.get(k) for k in ("total_cost_usd", "num_turns", "duration_ms", "is_error")}
    return info


# ---- many campaigns ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m aobench.agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="results and logs (same layout as the baselines)")
    p.add_argument("--work", type=Path, default=Path.home() / ".aobench" / "work",
                   help="agent workspaces; keep this outside the repository")
    p.add_argument("--tasks", nargs="*", default=list(SPECS), choices=list(SPECS))
    p.add_argument("--seeds", type=int, default=1, help="seeds 0..n-1")
    p.add_argument("--agent", default="claude", choices=["claude", "fake"])
    p.add_argument("--model")
    p.add_argument("--effort")
    p.add_argument("--workers", type=int, default=1, help="campaigns run at once")
    p.add_argument("--call-timeout", type=float, default=3600, help="seconds per agent call")
    p.add_argument("--max-calls", type=int, default=200, help="agent calls per campaign")
    a = p.parse_args(argv)
    if REPO in a.work.resolve().parents:
        p.error("--work must be outside the repository, so the agent cannot read the tasks")
    jobs = [(t, s) for t in a.tasks for s in range(a.seeds)
            if not (a.out / t / "agent" / f"seed_{s}.csv").exists()]
    print(f"{len(jobs)} campaigns to run", flush=True)

    def one(job):
        t, s = job
        t0 = time.time()
        c = Campaign(t, s, a.out, a.work, a.agent, a.model, a.effort, a.call_timeout, a.max_calls)
        try:
            reason = c.run()
        except Exception as e:
            reason = f"error: {e}"
        score = np.log10(max(c.regret, FLOOR)) if c.rows else float("nan")
        print(f"{t} seed {s}: {reason}, {len(c.rows)}/{c.budget} evaluations, "
              f"log10 regret {score:.2f}, {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, jobs))


if __name__ == "__main__":
    main()
