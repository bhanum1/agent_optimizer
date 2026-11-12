"""Launching the agent: a fresh, headless Claude Code (or the fake agent), and reading its log.

Claude logs in with the ANTHROPIC_* settings of the repository's .env (an API key,
or a gateway), or else with the claude.ai login of this machine, whose token is
refreshed outside the sandbox and passed in.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import REPO, skill_python

TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]  # file and shell tools only: no web
TOKEN_MARGIN = 15 * 60  # seconds a login token must still be valid for when a call starts
_token_lock = threading.Lock()


def agent_command(agent: str, model: str | None, effort: str | None) -> tuple[list[str], dict]:
    """The command for one agent call (it reads the prompt on stdin), and its environment."""
    if agent == "fake":
        return [sys.executable, "-m", "aobench.agent.fake"], {**os.environ, "AOPT_PYTHON": str(skill_python())}
    # No user settings, memory, skills or MCP servers, and no session kept between calls.
    command = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--no-session-persistence",
               "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
               "--tools", ",".join(TOOLS), "--allowedTools", *TOOLS, "--permission-mode", "dontAsk"]
    if model:
        command += ["--model", model]
    if effort:
        command += ["--effort", effort]
    env = {**claude_env(), "ENABLE_CLAUDEAI_MCP_SERVERS": "false", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)  # never an enclosing session's: this machine's login, or none
    if not env.get("ANTHROPIC_BASE_URL") and not env.get("ANTHROPIC_API_KEY"):
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token()
    return command, env


def claude_env() -> dict:
    """The environment for `claude -p`. The repository's .env overrides inherited ANTHROPIC_* values.

    With ANTHROPIC_BASE_URL set to a gateway, the key in ANTHROPIC_API_KEY is sent as a Bearer token
    (ANTHROPIC_AUTH_TOKEN), which is how such gateways authenticate. The login of an enclosing Claude
    Code session is dropped, or the child would send that instead of the key."""
    env = dict(os.environ)
    env.update({k: v for k, v in read_dotenv(REPO / ".env").items() if k.startswith("ANTHROPIC_")})
    base = env.get("ANTHROPIC_BASE_URL", "")
    if not base or "api.anthropic.com" in base:
        return env
    if "ANTHROPIC_API_KEY" in env:
        env["ANTHROPIC_AUTH_TOKEN"] = env.pop("ANTHROPIC_API_KEY")
    env = {k: v for k, v in env.items()
           if not (k.startswith("CLAUDE") or (k.startswith("USE_") and k.endswith("_OAUTH")))}
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    return env


def read_dotenv(path: Path) -> dict:
    """KEY=value lines of a .env file; an empty dict if there is none."""
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key and not key.startswith("#"):
                values[key.strip()] = value.strip().strip("'\"")
    return values


def oauth_token() -> str:
    """The claude.ai login's access token, refreshed (outside the sandbox) when it is near expiry."""
    creds = Path.home() / ".claude" / ".credentials.json"

    def read():
        o = json.loads(creds.read_text())["claudeAiOauth"]
        return o["accessToken"], o["expiresAt"] / 1000 - time.time()

    with _token_lock:
        token, left = read()
        if left < TOKEN_MARGIN:
            # `claude auth status` refreshes an expired token; one that is only near expiry is waited out.
            if left > 0:
                time.sleep(left + 5)
            subprocess.run(["claude", "auth", "status"], capture_output=True, timeout=120)
            token, left = read()
            if left < TOKEN_MARGIN:
                raise RuntimeError("Could not refresh the claude.ai login: run `claude auth login` here.")
        return token


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
