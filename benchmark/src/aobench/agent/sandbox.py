"""The bubblewrap sandbox the agent and its optimizers run in (Linux only).

Inside: the workspace at /work (read-write), the skill's Python read-only, the
system's /usr and /etc read-only, an empty home, and its own process table.
Nothing else of the host: not the repository, other workspaces, results or
~/.claude.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import skill_python

VENV_IN_BOX = "/opt/aopt-venv"
# Environment variables that cross into the sandbox, by prefix; everything else is left outside.
PASSED = ("ANTHROPIC_", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
          "ENABLE_CLAUDEAI_MCP_SERVERS", "CLAUDE_CODE_DISABLE_AUTO_MEMORY")


def available() -> bool:
    return shutil.which("bwrap") is not None


def sandboxed(workspace: Path, command: list[str], env: dict) -> tuple[list[str], dict]:
    """`command` wrapped to run in the sandbox, and the environment it gets."""
    if not available():
        raise RuntimeError("The agent runs in a bubblewrap sandbox: install bwrap (Linux only).")
    venv = skill_python().parent.parent
    box = ["bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
           "--hostname", "sandbox", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run", "--tmpfs", "/home",
           "--tmpfs", "/opt", "--dir", "/home/agent"]
    for d in ("bin", "lib", "lib64", "sbin"):
        p = Path("/") / d
        box += ["--symlink", os.readlink(p), str(p)] if p.is_symlink() else ["--ro-bind-try", str(p), str(p)]
    if Path("/run/systemd/resolve").is_dir():  # where /etc/resolv.conf points on systemd hosts
        box += ["--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve"]
    # The skill's Python: the venv, and the interpreter it links to, each read-only.
    box += ["--ro-bind", str(venv), VENV_IN_BOX]
    for p in _python_install_paths(venv):
        box += ["--ro-bind", str(p.resolve()), str(p)]
    box += ["--bind", str(workspace), "/work", "--chdir", "/work"]
    inner = {"PATH": "/opt/claude:/usr/bin:/bin", "HOME": "/home/agent", "USER": "agent", "LANG": "C.UTF-8",
             "TERM": "dumb", "AOPT_PYTHON": f"{VENV_IN_BOX}/bin/python", "DISABLE_AUTOUPDATER": "1"}
    if command and command[0] == "claude":
        box += ["--ro-bind", str(Path(shutil.which("claude")).resolve()), "/opt/claude/claude"]
        inner |= {k: v for k, v in env.items() if k.startswith(PASSED)}
        inner["CLAUDE_CONFIG_DIR"] = "/home/agent/.claude"  # empty: no settings, memory or transcripts
    return box + command, inner


def _python_install_paths(venv: Path) -> set[Path]:
    """The interpreter's install directories as the venv names them (they may be symlinks)."""
    link = venv / "bin" / "python"
    paths = {(link.parent / os.readlink(link)).parent.parent}  # absolute or relative to bin/
    for line in (venv / "pyvenv.cfg").read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "home":
            paths.add(Path(value.strip()).parent)
    return paths
