"""Shared campaign fixtures for the advisor-mode tests."""
import contextlib
import io
from pathlib import Path
import textwrap

PROBLEM = '''\
name = "test campaign"
[objective]
name = "yield"
direction = "maximize"
[budget]
total = 20
block_size = 4
[[parameters]]
name = "temperature"
low = 80
high = 200
[[parameters]]
name = "conc"
low = 0.001
high = 1.0
log = true
[[parameters]]
name = "stir"
type = "integer"
low = 1
high = 30
'''

HEADER = 'id,block,status,source,optimizer,temperature,conc,stir,yield,notes\n'


def campaign(folder, rows='', problem=PROBLEM, header=HEADER):
    """Write problem.toml and, if rows are given, data.csv; return the folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'problem.toml').write_text(problem)
    if rows:
        (folder / 'data.csv').write_text(header + textwrap.dedent(rows).lstrip())
    return folder


def quiet():
    """Silence a CLI call's stdout and stderr; returns (stack, stdout buffer)."""
    stack, out = contextlib.ExitStack(), io.StringIO()
    stack.enter_context(contextlib.redirect_stdout(out))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack, out
