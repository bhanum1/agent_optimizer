"""Run reports read a live journal read-only and classify run states."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agentoptimizer.blind_runner.evaluator import Evaluator
from agentoptimizer.blind_runner.runner import cli_environment
from scripts.agent_runs import progress, state, summary


class Noisy:
    dim = 2
    bounds = [[-5., 5.]] * 2
    def evaluate(self, x):
        # Observed y is off by one; scoring uses the noise-free value.
        return SimpleNamespace(y=sum(v * v for v in x) - 1., y_true=sum(v * v for v in x), elapsed=0.)


class Reports(unittest.TestCase):
    def campaign(self, tmp):
        root = Path(tmp) / 'run-000000000000'
        private = root / 'private'
        private.mkdir(parents=True)
        (private / 'manifest.json').write_text(json.dumps(
            {'task': 'fake2', 'budget': 5, 'seed': 0, 'agent': 'codex', 'model': 'm', 'reasoning_effort': 'low'}))
        (private / 'scoring.json').write_text(json.dumps({'task_name': 'fake', 'f_opt': 0.}))
        evaluator = Evaluator({'budget': 5, 'seed': 0, 'campaigns': [{'id': 'run'}]},
                              private / 'evaluations.sqlite', lambda _: Noisy())
        def call(op, **kwargs):
            reply = evaluator.handle('run', {'request_id': uuid.uuid4().hex, 'op': op, **kwargs})
            self.assertTrue(reply['ok'], reply)
        call('label', name='sobol', documents={'roadmap_initial.md': 'First block', 'changelog.md': ''})
        call('evaluate', x=[2., 0.])
        call('evaluate', x=[1., 0.])
        call('label', name='cma_es')
        call('evaluate', x=[3., 0.])
        return root, evaluator

    def test_progress_reads_live_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, evaluator = self.campaign(tmp)
            used, best, blocks = progress(root)
            self.assertEqual((used, best, blocks), (3, 1., [['sobol', 2], ['cma_es', 1]]))
            # The writer is still open and unaffected.
            evaluator.handle('run', {'request_id': uuid.uuid4().hex, 'op': 'evaluate', 'x': [0., 0.]})
            self.assertEqual(progress(root)[:2], (4, 0.))
            evaluator.db.close()
            self.assertEqual(summary(root)['gap'], 0.)

    def test_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, evaluator = self.campaign(tmp)
            evaluator.db.close()
            private = root / 'private'
            (private / 'launcher.json').write_text(json.dumps({'pid': os.getpid()}))
            self.assertEqual(state(root), 'running')
            ended = subprocess.Popen([sys.executable, '-c', ''])
            ended.wait()
            (private / 'launcher.json').write_text(json.dumps({'pid': ended.pid}))
            self.assertEqual(state(root), 'stopped')
            audit = {'unknown_evaluations': 0, 'artifacts_match_journal': True,
                     'only_workspace_execution_tools': True, 'tool_calls': 0}
            (private / 'integrity_audit.json').write_text(json.dumps(audit))
            self.assertEqual(state(root), 'check only')
            (private / 'cli_exit.json').write_text(json.dumps({'returncode': 0}))
            self.assertEqual(state(root), 'finished')
            (private / 'integrity_audit.json').write_text(json.dumps({**audit, 'unknown_evaluations': 1}))
            self.assertEqual(state(root), 'audit failed')


class CliEnvironment(unittest.TestCase):
    def test_parent_claude_session_is_dropped(self):
        session = {'CLAUDECODE': '1', 'CLAUDE_CODE_ENTRYPOINT': 'desktop', 'MCP_CONNECTION_NONBLOCKING': '1',
                   'ANTHROPIC_BASE_URL': 'http://localhost:1', 'PATH': '/usr/bin'}
        with patch.dict(os.environ, session, clear=True):
            self.assertEqual(cli_environment(), {'PATH': '/usr/bin'})
        terminal = {'CLAUDE_CONFIG_DIR': '/x', 'ANTHROPIC_BASE_URL': 'http://proxy', 'PATH': '/usr/bin'}
        with patch.dict(os.environ, terminal, clear=True):
            self.assertEqual(cli_environment(), terminal)


if __name__ == '__main__':
    unittest.main()
