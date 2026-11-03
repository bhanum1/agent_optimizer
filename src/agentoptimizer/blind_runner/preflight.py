"""Fail closed before inference if the native execution boundary leaks access."""
import json
import os
import shlex

from .execution import run


# argv is the forbidden paths (JSON), the supervisor's pid and a listening local
# port. The advisor has no evaluator access at all; the campaign folder is the
# only place it writes besides work/ and .tmp/.
PROBE = r'''
import errno, importlib.util, json, os, pathlib, socket, subprocess, sys
root = pathlib.Path.cwd()
checks = []
def denied(label, fn):
    try:
        fn()
    except OSError as e:
        # EBUSY, EXDEV: Linux bubblewrap protects files with read-only bind mounts, which cannot be
        # unlinked (EBUSY) or renamed/linked across (EXDEV, each mount is its own filesystem).
        assert e.errno in (errno.EACCES, errno.EPERM, errno.ENOENT, errno.EROFS, errno.EBUSY, errno.EXDEV), \
            (label, str(e))
        checks.append(label)
    else:
        raise AssertionError('Isolation failure: ' + label)
for value in json.loads(sys.argv[1]):
    denied('private read denied', lambda p=pathlib.Path(value): p.read_bytes())
denied('home configuration protected', lambda: (root/'.home/agent_config').write_text('override'))
denied('project configuration protected', lambda: (root/'.codex').mkdir())
link = root/'work/escape'
link.symlink_to(json.loads(sys.argv[1])[0])
denied('symlink escape denied', lambda: link.read_bytes())
link.unlink()
denied('hardlink escape denied', lambda: os.link(json.loads(sys.argv[1])[0], root/'work/hardlink'))
def connect(address):
    with socket.socket() as s:
        s.settimeout(1)
        s.connect(address)
denied('external network denied', lambda: connect(('1.1.1.1', 443)))
denied('local service denied', lambda: connect(('127.0.0.1', int(sys.argv[3]))))
if sys.platform == 'darwin':
    try:
        p = subprocess.run(['/bin/ps', '-p', sys.argv[2], '-o', 'command='], capture_output=True, text=True)
        assert p.returncode != 0 or not p.stdout.strip(), 'Supervisor process arguments exposed'
    except PermissionError:
        pass
    checks.append('supervisor process inaccessible')
assert not (root/'objective.py').exists() and not (root/'src/tasks').exists()
assert not (root/'.git').exists()
assert importlib.util.find_spec('ioh') is None, 'Use an optimizer-only runtime without ioh'
assert importlib.util.find_spec('cocoex') is None, 'Use an optimizer-only runtime without cocoex'
assert importlib.util.find_spec('agentoptimizer') is None, 'Use a runtime without the agentoptimizer package'
checks.append('objective sources and backends absent')
import numpy, scipy, cma, torch, botorch, gpytorch, pybobyqa
for name in ['AGENTS.md', 'opt_run.py', 'lib/agentoptimizer/advisor/cli.py']:
    denied(name + ' protected', lambda p=root/name: p.open('a'))
denied('campaign replacement denied', lambda: (root/'campaign').rename(root/'work/moved_campaign'))
denied('library replacement denied', lambda: (root/'lib').rename(root/'work/moved_lib'))
assert not (root/'lib/agentoptimizer/tasks').exists()
probe = root/'campaign/.write_probe'
probe.write_text('probe')
probe.unlink()
checks.append('campaign writable')
result = subprocess.run([sys.executable, '-I', 'opt_run.py', 'check', 'campaign'], capture_output=True, text=True)
assert result.returncode == 0, result.stdout + result.stderr
checks.append('advisor runs on the campaign')
print(json.dumps({'checks': checks, 'evaluations': 0}))
'''


def check(policy, forbidden, private):
    import socket
    # A listening socket distinguishes actual denial from "nothing listening".
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        script = '.venv/bin/python -B -c ' + shlex.quote(PROBE)
        script += ' ' + shlex.quote(json.dumps([str(p) for p in forbidden]))
        script += ' ' + str(os.getpid()) + ' ' + str(listener.getsockname()[1])
        result = run(policy, script, timeout=120)
    (private / 'preflight.stdout').write_text(result.stdout)
    (private / 'preflight.stderr').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError('Isolation preflight failed. See ' + str(private / 'preflight.stderr')
                           + '\n' + result.stdout[-3000:] + result.stderr[-3000:])
    report = json.loads(result.stdout.splitlines()[-1])
    (private / 'isolation_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
