"""Advisor replay: run advisor mode as a blind benchmark campaign.

The runner plays the lab. Each block, a fresh advisor call in the sandbox
proposes the next block in workspace/campaign/ exactly as /opt-run does; the
runner checks it, reproduces it, evaluates it privately and records the
results. See docs/plan/05-advisor-mode.md, "Phase 4 design".
"""
from pathlib import Path
import shutil

PACKAGE = Path(__file__).resolve().parents[1]

OPT_RUN = '''"""Advisor mode: manage the campaign folder. See --help."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from agentoptimizer.advisor.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
'''


def install_advisor(workspace):
    """Copy the advisor and batch code into workspace/lib, with an opt_run.py beside it.
    Both use only relative imports and the agent runtime's libraries, and name no task."""
    workspace = Path(workspace)
    library = workspace / 'lib/agentoptimizer'
    library.mkdir(parents=True)
    (library / '__init__.py').write_text('')
    for name in ['advisor', 'batch']:
        shutil.copytree(PACKAGE / name, library / name, ignore=shutil.ignore_patterns('__pycache__'))
    (workspace / 'opt_run.py').write_text(OPT_RUN)
