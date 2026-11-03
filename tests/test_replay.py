"""Advisor replay: the sandbox copy of the advisor, and the runner's checks."""
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
import uuid

from advisor_fixtures import campaign
from agentoptimizer.blind_runner.evaluator import Evaluator
from agentoptimizer.blind_runner.replay import install_advisor

REPO = Path(__file__).resolve().parents[1]
AGENT_PYTHON = REPO / '.agent-venv/bin/python'

PROPOSE = '''from agentoptimizer import batch
from agentoptimizer.advisor import Campaign
campaign = Campaign.load(__file__)
campaign.write(batch.space_filling(**campaign.inputs(), q=4, seed=1))
'''


@unittest.skipUnless(AGENT_PYTHON.exists(), 'needs .agent-venv')
class AdvisorCopy(unittest.TestCase):
    def test_copy_runs_a_block_in_the_agent_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            install_advisor(workspace)
            campaign(workspace / 'campaign')

            def opt_run(*args):
                return subprocess.run([str(AGENT_PYTHON), '-I', 'opt_run.py', *args], cwd=workspace,
                                      capture_output=True, text=True)
            self.assertEqual(opt_run('new-block', 'campaign').returncode, 0)
            (workspace / 'campaign/blocks/001/propose.py').write_text(PROPOSE)
            result = opt_run('run', 'campaign')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            lines = (workspace / 'campaign/blocks/001/proposals.csv').read_text().splitlines()
            self.assertEqual(len(lines), 5)
            # The copy holds no objectives, and the agent runtime has no package to fall back on.
            self.assertFalse((workspace / 'lib/agentoptimizer/tasks').exists())
            probe = subprocess.run([str(AGENT_PYTHON), '-I', '-c', 'import agentoptimizer'],
                                   capture_output=True, text=True)
            self.assertNotEqual(probe.returncode, 0)


class Documents(unittest.TestCase):
    def test_replay_decisions_are_append_only(self):
        class Task:
            dim = 2
            bounds = [[-1., 1.]] * 2

            def evaluate(self, x):
                return SimpleNamespace(y=sum(v * v for v in x), elapsed=0.)

        with tempfile.TemporaryDirectory() as tmp:
            suite = {'budget': 3, 'campaigns': [{'id': 'run'}], 'documents': {'decisions.md': 'append'}}
            e = Evaluator(suite, Path(tmp) / 'state.sqlite', lambda _: Task())

            def call(op, **kwargs):
                return e.handle('run', {'request_id': uuid.uuid4().hex, 'op': op, **kwargs})
            self.assertTrue(call('label', name='block 1 sobol',
                                 documents={'decisions.md': '## Block 1\n'})['ok'])
            self.assertTrue(call('evaluate', x=[0., 0.])['ok'])
            self.assertTrue(call('label', name='block 2 gp',
                                 documents={'decisions.md': '## Block 1\n## Block 2\n'})['ok'])
            reply = call('label', name='block 3', documents={'decisions.md': '## Block 1 (edited)\n'})
            self.assertFalse(reply['ok'])
            self.assertIn('append-only', reply['error'])
            reply = call('label', name='x', documents={'roadmap_initial.md': '', 'changelog.md': ''})
            self.assertFalse(reply['ok'])
            e.db.close()


if __name__ == '__main__':
    unittest.main()
