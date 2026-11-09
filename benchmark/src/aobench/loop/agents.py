"""Launch one skill call: a Codex or Claude CLI whose only tool is the
sandboxed run_command. With cfg["context"] == "fresh" (the default) each call
starts with no memory of earlier calls. With "persistent" (Claude only) every
call resumes one session, so the agent keeps its earlier reasoning and tool output.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from .sandbox import overrides
from .tools import MAX_SECONDS

TOOLS = Path(__file__).with_name("tools.py")
SERVER = {"codex": "workspace", "claude": "sandbox"}  # Claude Code reserves "workspace"
FRESH = "You start every call with no memory of earlier calls; what you need to remember goes in `agent/context.md`."
PERSISTENT = ("This conversation continues across calls: you remember your earlier calls, their analysis and "
              "tool output. Still keep `agent/context.md` current; it is part of the record.")


def codex_command(cfg, workspace: Path, private: Path, system: str):
    config = dict(cfg["policy"]["agent"])
    config.update({
        "developer_instructions": system,
        "model_reasoning_effort": cfg["effort"],
        "mcp_servers": {"workspace": {
            "command": sys.executable, "args": ["-B", str(TOOLS), str(private / "policy.json")],
            "enabled": True, "required": True, "startup_timeout_sec": 30,
            "tool_timeout_sec": MAX_SECONDS + 30, "enabled_tools": ["run_command"],
            "default_tools_approval_mode": "approve"}},
    })
    for feature in ["shell_tool", "unified_exec", "view_image", "apps", "plugins", "remote_plugin",
                    "memories", "hooks", "shell_snapshot", "browser_use", "browser_use_external",
                    "computer_use", "in_app_browser", "multi_agent", "multi_agent_v2",
                    "image_generation", "skill_search", "workspace_dependencies", "code_mode"]:
        config["features." + feature] = False
    config["features.code_mode_host"] = True
    return [cfg["codex"], "--no-daemon", "exec", "--ignore-user-config", "--ignore-rules",
            "--ephemeral", "--skip-git-repo-check", "--json", "--color", "never",
            "-m", cfg["model"], "-C", str(workspace), *overrides(config), "-"], dict(os.environ)


def claude_command(cfg, workspace: Path, private: Path, system: str, number: int = 1):
    # No built-in tools, settings, memory, skills or connectors: only run_command.
    mcp = {"mcpServers": {"sandbox": {"type": "stdio", "command": sys.executable,
                                      "args": ["-B", str(TOOLS), str(private / "policy.json")]}}}
    (private / "claude_mcp.json").write_text(json.dumps(mcp))
    command = [cfg["claude"], "-p", "--output-format", "stream-json", "--verbose",
               "--model", cfg["model"], "--effort", cfg["effort"], "--tools", "",
               "--strict-mcp-config", "--mcp-config", str(private / "claude_mcp.json"),
               "--allowedTools", "mcp__sandbox__run_command",
               "--permission-mode", "dontAsk", "--setting-sources", "", "--disable-slash-commands",
               "--system-prompt", system]
    if cfg.get("context", "fresh") == "persistent":
        # One session per campaign: the first call creates it, later calls resume it.
        session = private / "session_id.txt"
        if number == 1:
            session.write_text(str(uuid.uuid4()))
            command += ["--session-id", session.read_text()]
        else:
            command += ["--resume", session.read_text()]
    else:
        command.append("--no-session-persistence")
    env = {**os.environ, "ENABLE_CLAUDEAI_MCP_SERVERS": "false", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
           "GIT_CEILING_DIRECTORIES": str(workspace.parent), "MCP_TOOL_TIMEOUT": str((MAX_SECONDS + 30) * 1000)}
    return command, env


def call(cfg, workspace: Path, private: Path, number: int, prompt: str) -> int:
    """Run skill call `number`; logs go to private/calls/<number>/. Returns the CLI exit code."""
    system = (workspace / "skill" / "METHOD.md").read_text()
    if cfg.get("context", "fresh") == "persistent":
        if cfg["agent"] != "claude":
            raise ValueError("Persistent context is implemented for the Claude agent only.")
        if FRESH not in system:
            raise ValueError("METHOD.md no longer contains the fresh-context sentence to replace.")
        system = system.replace(FRESH, PERSISTENT)
        command, env = claude_command(cfg, workspace, private, system, number)
    else:
        build = codex_command if cfg["agent"] == "codex" else claude_command
        command, env = build(cfg, workspace, private, system)
    logs = private / "calls" / f"{number:03d}"
    logs.mkdir(parents=True)
    (logs / "prompt.txt").write_text(prompt)
    with open(logs / "events.jsonl", "w") as events, open(logs / "stderr.log", "w") as errors:
        process = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                                   stdout=events, stderr=errors, text=True)
        try:
            process.communicate(prompt, timeout=cfg["call_timeout"])
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return -9
        except BaseException:
            process.kill()
            process.wait()
            raise
    return process.returncode


def tool_calls(events: Path, agent: str) -> list[tuple[str, str]]:
    """(server, tool) for every tool use in one call's event log; anything
    other than the sandbox's run_command is reported as ("unexpected", name)."""
    out = []
    for line in events.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        item = event.get("item", {}) if event.get("type") == "item.completed" else {}
        if item.get("type") == "mcp_tool_call":
            out.append((item.get("server"), item.get("tool")))
        elif item.get("type") in ("command_execution", "file_change", "web_search"):
            out.append(("unexpected", item["type"]))
        elif event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    _, server, tool = (block.get("name", "").split("__", 2) + ["", ""])[:3]
                    out.append((server, tool))
                elif block.get("type") == "server_tool_use":
                    out.append(("unexpected", block.get("name")))
    return [(s, t) if (s, t) == (SERVER[agent], "run_command") else ("unexpected", f"{s}:{t}")
            for s, t in out]
