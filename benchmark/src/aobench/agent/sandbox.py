"""The sandbox the agent and its optimizers run in: bubblewrap on Linux, Seatbelt on macOS.

Linux (bubblewrap): the workspace at /work (read-write), the skill's Python
read-only, the system's /usr and /etc read-only, an empty home, and its own
process table. Nothing else of the host: not the repository, other workspaces,
results or ~/.claude.

macOS (sandbox-exec, Seatbelt): the agent sees the real paths, but cannot read
or write anything in the user's home or /private/tmp except the workspace, its
own empty home beside it, the skill's Python and the claude binary. It cannot
list or inspect other processes. Paths elsewhere in the home can be stat'ed but
not read or listed, so git finds no repository and the repository's files
(the task definitions, results) stay unreadable.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from . import skill_python

VENV_IN_BOX = "/opt/aopt-venv"
# Environment variables that cross into the sandbox, by prefix; everything else is left outside.
PASSED = ("ANTHROPIC_", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
          "ENABLE_CLAUDEAI_MCP_SERVERS", "CLAUDE_CODE_DISABLE_AUTO_MEMORY")
MACOS = sys.platform == "darwin"

SEATBELT = """(version 1)
(allow default)
(deny process-info*)
(allow process-info* (target same-sandbox))
(deny file-read-data file-write* (subpath (param "HOME")) (subpath "/private/tmp") (subpath "/Volumes"))
(allow file-read-data file-write* (subpath (param "WORK")) (subpath (param "BOXHOME")))
(allow file-read-data {readable})
"""


def available() -> bool:
    return shutil.which("sandbox-exec" if MACOS else "bwrap") is not None


def root(workspace: Path) -> Path:
    """The workspace's path as seen inside the sandbox."""
    return workspace if MACOS else Path("/work")


def home(workspace: Path) -> Path:
    """The agent's empty home on macOS: beside the workspace, so the agent's own files stay out of it."""
    return workspace.with_name(workspace.name + ".home")


def sandboxed(workspace: Path, command: list[str], env: dict) -> tuple[list[str], dict]:
    """`command` wrapped to run in the sandbox, and the environment it gets."""
    if not available():
        raise RuntimeError("The agent runs in a sandbox: bubblewrap (bwrap) on Linux, sandbox-exec on macOS.")
    return (_seatbelt if MACOS else _bwrap)(workspace, command, env)


def _seatbelt(workspace: Path, command: list[str], env: dict) -> tuple[list[str], dict]:
    workspace, box_home = workspace.resolve(), home(workspace).resolve()
    (box_home / "tmp").mkdir(parents=True, exist_ok=True)
    python = skill_python()
    venv = python.parent.parent
    readable = {venv.resolve(), *(p.resolve() for p in _python_install_paths(venv))}
    inner = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(box_home), "USER": "agent", "LANG": "C.UTF-8",
             "TERM": "dumb", "TMPDIR": str(box_home / "tmp"), "AOPT_PYTHON": str(python),
             "TMPPREFIX": str(box_home / "tmp" / "zsh"),  # zsh heredocs use /tmp/zsh otherwise, ignoring TMPDIR
             "DISABLE_AUTOUPDATER": "1", "GIT_CEILING_DIRECTORIES": str(workspace.parent)}
    if command and command[0] == "claude":
        claude = Path(shutil.which("claude")).resolve()
        readable.add(claude)
        command = [str(claude), *command[1:]]
        inner |= {k: v for k, v in env.items() if k.startswith(PASSED)}
        inner["CLAUDE_CONFIG_DIR"] = str(box_home / ".claude")  # empty: no settings, memory or transcripts
        inner["CLAUDE_CODE_TMPDIR"] = inner["TMPDIR"]  # else it uses /tmp/claude-<uid>, shared and unreadable here
    profile = SEATBELT.format(readable=" ".join(f'(subpath "{p}")' for p in sorted(readable)))
    box = ["sandbox-exec", "-D", f"HOME={Path.home().resolve()}", "-D", f"WORK={workspace}",
           "-D", f"BOXHOME={box_home}", "-p", profile]
    return box + command, inner


def _bwrap(workspace: Path, command: list[str], env: dict) -> tuple[list[str], dict]:
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
