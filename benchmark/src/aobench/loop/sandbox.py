"""Native sandbox policies for agent tools and optimizer hosts.

Both run under `codex sandbox -P blind`: no network, the workspace readable,
and only a few workspace directories writable. The agent may write
.agentoptimizer/work/, .agentoptimizer/agent/ and .tmp/; an optimizer host may
write .agentoptimizer/state/ and .tmp/. Neither can
read private/ or the objective.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

AGENT_WRITES = [".agentoptimizer/work", ".agentoptimizer/agent", ".tmp"]
HOST_WRITES = [".agentoptimizer/state", ".tmp"]
BACKENDS = ["aobench", "ioh", "cocoex", "opfunu", "pymoo", "yahpo_gym", "nevergrad"]


def toml(value):
    if isinstance(value, dict):
        return "{" + ", ".join(json.dumps(k) + "=" + toml(v) for k, v in value.items()) + "}"
    return json.dumps(value)


def overrides(config: dict) -> list[str]:
    return [part for key, value in config.items() for part in ("-c", key + "=" + toml(value))]


def environment(workspace: Path) -> dict:
    return {
        "PATH": str(workspace / ".venv/bin") + ":/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": str(workspace / ".tmp"), "HOME": str(workspace / ".home"),
        "XDG_CACHE_HOME": str(workspace / ".tmp/cache"), "MPLCONFIGDIR": str(workspace / ".tmp/mpl"),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }


def permissions(workspace: Path, runtime_roots, writable) -> dict:
    # No ":tmpdir" entry: TMPDIR is workspace/.tmp, granted below; a deny there would win.
    files = {":root": "deny", ":minimal": "read", ":slash_tmp": "deny"}
    files.update({str(path): "read" for path in runtime_roots})
    files[str(workspace)] = "read"
    for path in writable:
        files[str(workspace / path)] = "write"
    return {
        "default_permissions": "blind",
        "permissions.blind.filesystem": files,
        "permissions.blind.network.enabled": False,
        "approval_policy": "never", "allow_login_shell": False,
        "shell_environment_policy.inherit": "none",
        "web_search": "disabled", "project_doc_max_bytes": 0,
        "memories.use_memories": False, "memories.generate_memories": False,
        "mcp_servers": {}, "features.skip_host_skill_discovery": True,
    }


def prefix(policy: dict, which: str) -> list[str]:
    """Command prefix that runs argv inside the sandbox; `which` is "agent" or "host"."""
    return [policy["codex"], "--no-daemon", *overrides(policy[which]),
            "sandbox", "-P", "blind", "-C", policy["workspace"], "--"]


def runtime_info(python: Path) -> dict:
    """Prefix and versions of the optimizer-only Python; refuses one with objective backends."""
    query = f"""import sys, json, importlib.util, importlib.metadata as m
names = ['numpy', 'scipy', 'cma', 'botorch', 'gpytorch', 'torch', 'pandas']
print(json.dumps({{'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'executable': sys.executable,
    'version': sys.version, 'backends': [n for n in {BACKENDS!r} if importlib.util.find_spec(n)],
    'packages': {{n: m.version(n) for n in names if importlib.util.find_spec(n.lower())}}}}))
"""
    info = json.loads(subprocess.check_output([str(python), "-I", "-c", query], text=True))
    if info["backends"]:
        raise ValueError("Agent runtime contains objective backends: " + ", ".join(info["backends"])
                         + ". Build it with `UV_PROJECT_ENVIRONMENT=.agent-venv uv sync --project skill/agentoptimizer --no-dev --locked`.")
    return info


def link_venv(workspace: Path, runtime: dict) -> None:
    """A read-only venv view at workspace/.venv pointing at the optimizer-only runtime."""
    venv = workspace / ".venv"
    (venv / "bin").mkdir(parents=True)
    executable = Path(runtime["executable"]).resolve()
    (venv / "bin/python").symlink_to(executable)
    (venv / "pyvenv.cfg").write_text(f"home = {executable.parent}\ninclude-system-site-packages = false\n")
    (venv / "lib").symlink_to(Path(runtime["prefix"]).resolve() / "lib", target_is_directory=True)
    if (Path(runtime["prefix"]) / "lib64").exists():
        (venv / "lib64").symlink_to("lib", target_is_directory=True)


def runtime_roots(runtime: dict, codex: Path) -> list[str]:
    import sys
    roots = {runtime["prefix"], runtime["base_prefix"],
             str(Path(runtime["prefix"]).resolve()), str(Path(runtime["base_prefix"]).resolve())}
    if sys.platform.startswith("linux"):
        # Codex's Linux sandbox re-executes the codex binary inside bubblewrap.
        roots.add(str(Path(codex).resolve().parents[1]))
    return sorted(roots)
