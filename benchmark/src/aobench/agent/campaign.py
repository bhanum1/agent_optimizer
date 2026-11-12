"""One campaign: a task, a seed, and the agent called block after block until the budget is spent."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..tasks import make_task
from . import SKILL, sandbox, skill_python
from .claude import agent_command, final_event, transcript
from .problem import problem_md, problem_yaml

# Copied into each workspace: the skill as a user installs it, without its development files.
SKILL_IGNORE = shutil.ignore_patterns(".venv", "uv.lock", "tests", "install.sh", "__pycache__", ".DS_Store",
                                      "*.egg-info", ".pytest_cache")
# `aopt suggest` output that is not news: everything else it prints is a message about the block.
ROUTINE = ("Requested", "Still waiting", " ", "Measure them", "The budget is spent", "Block ", "Waiting:")
WAITING = "Waiting:"  # `aopt suggest --serve` asks for the pending rows to be measured
MAX_FAILURES = 3  # consecutive calls that yield no evaluation before the campaign stops

PROMPT = """You are deciding the next block of an optimization campaign. The skill is in `skill/`: \
read `skill/METHOD.md` and follow it. The project folder is `.agentoptimizer/`, and `aopt` is `skill/aopt`, \
run from this directory. {used} of {budget} measurements are used; {remaining} remain. When `aopt validate` \
accepts your proposal, stop: the harness commits the block and measures its points, then calls you again.{note}"""


class Campaign:
    def __init__(self, task_key: str, seed: int, out: Path, work: Path, agent: str, model: str | None,
                 effort: str | None, call_timeout: float, max_calls: int):
        self.task = make_task(task_key, seed=seed)
        self.budget = self.task.spec.budget
        self.result = out / task_key / "agent" / f"seed_{seed}.csv"
        self.logs = out / task_key / "agent" / f"seed_{seed}"
        # An opaque name: the host path appears in the sandbox command line, which the agent can read.
        key = f"{out.resolve()}|{task_key}|{seed}".encode()
        self.workspace = work.resolve() / hashlib.sha256(key).hexdigest()[:16]
        self.project = self.workspace / ".agentoptimizer"
        self.agent, self.model, self.effort = agent, model, effort
        self.sandboxed = agent == "claude"
        self.call_timeout, self.max_calls = call_timeout, max_calls
        self.rows: list[tuple] = []  # eval, y, regret: what the result CSV holds
        self.best_y, self.regret = np.inf, np.inf

    def prepare(self) -> None:
        for d in (self.workspace, sandbox.home(self.workspace), self.logs):
            if d.exists():
                shutil.rmtree(d)
        self.logs.mkdir(parents=True)
        (self.logs / "workspace.txt").write_text(f"{self.workspace}\n")
        shutil.copytree(SKILL, self.workspace / "skill", ignore=SKILL_IGNORE)
        self.project.mkdir()
        (self.project / "problem.yaml").write_text(problem_yaml(self.task))
        (self.project / "PROBLEM.md").write_text(problem_md(self.task))
        self.aopt("init")

    def aopt_command(self, *args) -> tuple[list[str], dict]:
        # In the sandbox too: `aopt suggest` runs the agent's optimizer code.
        if self.sandboxed:
            aopt = sandbox.root(self.workspace) / "skill" / "aopt"
            return sandbox.sandboxed(self.workspace, [str(aopt), *args], {})
        return ([str(self.workspace / "skill" / "aopt"), *args],
                {**os.environ, "AOPT_PYTHON": str(skill_python())})

    def aopt(self, *args, check=True) -> subprocess.CompletedProcess:
        command, env = self.aopt_command(*args)
        out = subprocess.run(command, cwd=self.workspace, env=env, capture_output=True, text=True)
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
        if self.sandboxed:
            command, env = sandbox.sandboxed(self.workspace, command, env)
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
        log = transcript(logs / "events.jsonl", prompt)
        (logs / "transcript.md").write_text(log)
        with open(self.logs / "transcript.md", "a") as f:  # the whole campaign's reasoning, call after call
            f.write(f"# Call {number}\n\n{log}\n")
        return {"exit": code, **final_event(logs / "events.jsonl")}

    def run_block(self) -> tuple[int, list[str]]:
        """Work through the committed block as a user would. Returns evaluations made and aopt's messages.

        One `aopt suggest --serve` for the whole block, so the optimizer stays in memory: after each
        request it waits for the pending rows to be measured and a line on its stdin.
        """
        used, messages = 0, []
        command, env = self.aopt_command("suggest", "--serve")
        log = self.logs / "serve.log"  # aopt's stderr, block after block
        start = log.stat().st_size if log.exists() else 0
        with open(log, "a") as errors:
            process = subprocess.Popen(command, cwd=self.workspace, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    line = line.rstrip("\n")
                    if line and not line.startswith(ROUTINE):
                        messages.append(line)
                    if not line.startswith(WAITING):
                        continue
                    n = self.measure_pending()
                    used += n
                    if n == 0:
                        break  # the budget is spent: stop, and aopt saves the optimizer
                    process.stdin.write("\n")
                    process.stdin.flush()
                process.stdin.close()
                process.wait(timeout=600)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        if process.returncode:
            with open(log) as f:
                f.seek(start)
                error = f.read().strip()[-4000:]
            messages.append(error or f"aopt suggest exited with code {process.returncode}")
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
