"""Fail closed before any model call if either sandbox policy leaks access."""
from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

from .sandbox import environment, prefix

PROBE = r'''
import errno, importlib.util, json, pathlib, socket, sys
root = pathlib.Path.cwd()
forbidden, port, allowed, protected = json.loads(sys.argv[1])
checks = []
def denied(label, fn):
    try:
        fn()
    except OSError as e:
        # EBUSY/EXDEV: Linux bubblewrap protects files with read-only bind mounts.
        assert e.errno in (errno.EACCES, errno.EPERM, errno.ENOENT, errno.EROFS, errno.EBUSY, errno.EXDEV), (label, str(e))
        checks.append(label)
    else:
        raise AssertionError("Isolation failure: " + label)
for value in forbidden:
    denied("read denied: " + value, lambda p=pathlib.Path(value): p.read_bytes())
for name in protected:
    p = root / name
    denied("write denied: " + name, (lambda p=p: p.open("a")) if p.is_file() else (lambda p=p: (p / "probe").write_text("x")))
for name in allowed:
    (root / name / "probe").write_text("x")
    (root / name / "probe").unlink()
    checks.append("write allowed: " + name)
link = root / allowed[0] / "escape"
link.symlink_to(forbidden[0])
denied("symlink escape denied", lambda: link.read_bytes())
link.unlink()
def connect(address):
    with socket.socket() as s:
        s.settimeout(1)
        s.connect(address)
denied("external network denied", lambda: connect(("1.1.1.1", 443)))
denied("local service denied", lambda: connect(("127.0.0.1", port)))
for name in ["ioh", "cocoex", "opfunu", "pymoo", "yahpo_gym", "nevergrad"]:
    assert importlib.util.find_spec(name) is None, "objective backend importable: " + name
checks.append("objective backends absent")
print(json.dumps(checks))
'''

FIXED = ["SKILL.md", "PROBLEM.md", "space.json", "host.py", "check.py", "data.csv", "decisions.md", "lib", "blocks"]


def check(policy: dict, forbidden: list[Path], private: Path) -> dict:
    workspace = Path(policy["workspace"])
    cases = {"agent": (["work", "agent", ".tmp"], FIXED + ["state"]),
             "host": (["state", ".tmp"], FIXED + ["work", "agent"])}
    report = {}
    # A listening socket distinguishes actual denial from "nothing listening".
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        for which, (allowed, protected) in cases.items():
            arg = json.dumps([[str(p) for p in forbidden], listener.getsockname()[1], allowed, protected])
            result = subprocess.run([*prefix(policy, which), ".venv/bin/python", "-I", "-B", "-c", PROBE, arg],
                                    cwd=workspace, env=environment(workspace), capture_output=True,
                                    text=True, timeout=120)
            (private / f"preflight_{which}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"Isolation preflight ({which}) failed:\n" + (result.stdout + result.stderr)[-3000:])
            report[which] = json.loads(result.stdout.splitlines()[-1])
    (private / "isolation_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
