"""The agent harness: run the skill on a benchmark task, with the harness as its user.

    python -m aobench.agent --out benchmark/runs/agent --tasks gallagher --seeds 1
    python -m aobench.agent --out benchmark/runs/try --tasks gallagher --agent fake

One campaign, for one task and seed (campaign.py):
1. Make a fresh workspace with a copy of the skill, write the problem definition
   (problem.py: `problem.yaml` and `PROBLEM.md`), and run `aopt init`.
2. Call the agent: a fresh, headless `claude -p` in the workspace (claude.py),
   told to follow the skill's METHOD.md. It writes a proposal: an optimizer of
   its own, or a list of points.
3. `aopt commit`, then `aopt suggest`; evaluate the task at every pending row of
   `data.csv`, write the values in, and `aopt suggest` again, until the block is
   done. This is exactly what a user does by hand.
4. Repeat from 2 until the budget is spent.

The harness talks to the skill only through `aopt` and the project files, as a
user would; it imports nothing from the skill.

The task is a black box. The problem definition names no function: variables
x0.., objective y, a neutral project name. The agent and every `aopt` call (which
runs the agent's own optimizer code) run in a sandbox (sandbox.py): bubblewrap on
Linux, Seatbelt (sandbox-exec) on macOS. It lets them read the workspace, the
skill's Python and an empty home, and nothing else: not the repository, the task
definitions, other workspaces or results, even though the workspaces are stored
inside the run folder. The fake agent (fake.py) runs without a sandbox.

Layout, per campaign:

    <out>/workspaces/<id>/      the workspace (/work in the Linux sandbox); --work moves it. The id is
                                opaque, since the sandbox's command line is visible inside it
      skill/                    copy of the skill folder
      .agentoptimizer/          the project: problem.yaml, data.csv, decisions.md, agent/, work/, ...
    <out>/workspaces/<id>.home/ the agent's empty home and temp folder (macOS only)
    <out>/<task>/agent/seed_<s>.csv            eval, y, regret: the same format as the baselines
    <out>/<task>/agent/seed_<s>/workspace.txt  the path of the campaign's workspace
    <out>/<task>/agent/seed_<s>/calls.jsonl    one line per agent call: blocks, cost, time
    <out>/<task>/agent/seed_<s>/transcript.md  every call's reasoning, messages, tool calls and results
    <out>/<task>/agent/seed_<s>/calls/NNN/     prompt, event stream, transcript.md, stderr, the agent's outputs

Finished campaigns (their CSV exists) are skipped. An unfinished one is not
resumed: its workspace is removed and it starts again.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SKILL = REPO / "skill"


def skill_python() -> Path:
    """The skill's own interpreter, built with uv on first use. Every `aopt` call runs with it."""
    python = SKILL / ".venv" / "bin" / "python"
    if not python.exists():
        subprocess.run(["uv", "sync", "--project", str(SKILL)], check=True)
    return python
