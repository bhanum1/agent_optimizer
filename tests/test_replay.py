"""Advisor replay: the sandbox copy of the advisor, and the runner's checks."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid

import numpy as np

from advisor_fixtures import campaign
from agentoptimizer.advisor.blocks import accept_block, new_block, run_block
from agentoptimizer.advisor.data import read_table
from agentoptimizer.advisor.problem import load_problem
from agentoptimizer.blind_runner.evaluator import Evaluator
from agentoptimizer.blind_runner.replay import (Replay, ReplayError, build_campaign, install_advisor, problem_toml,
                                                system_prompt, usage_of)

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



class Sphere:
    dim, noisy, pool, n_constraints = 2, False, False, 0
    bounds = [[-2., 2.], [-1., 3.]]

    def evaluate(self, x):
        return SimpleNamespace(y=sum(v * v for v in x), elapsed=0.)


class Pool:
    noisy, pool, indexed_pool, n_constraints = False, True, True, 0

    def __init__(self):
        self.candidates = np.array([[0., 1.], [1., 0.], [2., 2.], [3., 1.], [1., 1.], [0., 0.]])
        self.n_candidates, self.dim = len(self.candidates), 2
        self.bounds = np.column_stack((self.candidates.min(0), self.candidates.max(0)))

    def validate_index(self, index):
        if not 0 <= index < self.n_candidates:
            raise ValueError('bad index')

    def evaluate_index(self, index):
        return SimpleNamespace(y=float(self.candidates[index].sum()), elapsed=0.)


SCRIPT = '''from agentoptimizer import batch
from agentoptimizer.advisor import Campaign
campaign = Campaign.load(__file__)
campaign.write(batch.{method}(**campaign.inputs(), q=min(4, campaign.remaining), seed={seed}))
'''


def scripted(public, tamper=None):
    """An advisor without a model: /opt-run's steps with fixed choices.
    tamper(block, attempt) may change the campaign after the block is accepted."""
    def advise(block, attempt, logs):
        with contextlib.redirect_stdout(io.StringIO()):
            number, folder = new_block(public)
            with (public / 'decisions.md').open('a') as stream:
                stream.write(f'## Block {number}\nSpace filling.\n')
            method = 'space_filling' if number == 1 else 'random'
            (folder / 'propose.py').write_text(SCRIPT.format(method=method, seed=number))
            run_block(public, number, verify=False)
            accept_block(public, number)
        if tamper:
            tamper(block, attempt)
        return {'turns': 1, 'cost_usd': .5}
    return advise


class Loop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def replay(self, task, budget=10, tamper=None, advise=None):
        workspace, private = build_campaign(Path(self.tmp.name) / uuid.uuid4().hex, task, budget, 4, seed=3)
        suite = {'budget': budget, 'campaigns': [{'id': 'run'}], 'documents': {'decisions.md': 'append'}}
        self.evaluator = Evaluator(suite, private / 'evaluations.sqlite', lambda _: task)
        self.addCleanup(self.evaluator.db.close)
        public = workspace / 'campaign'

        def sandbox(command):
            return subprocess.run(['/bin/bash', '-c', command], cwd=workspace, capture_output=True, text=True)
        return Replay(workspace, private, self.evaluator, advise or scripted(public, tamper), sandbox,
                      python=sys.executable), public

    def test_three_blocks_use_the_budget_and_match_the_journal(self):
        replay, public = self.replay(Sphere())
        records = replay.run()
        self.assertEqual([(r['block'], r['rows'], r['status']) for r in records],
                         [(1, 4, 'evaluated'), (2, 4, 'evaluated'), (3, 2, 'evaluated')])
        self.assertEqual(records[0]['methods'], 'sobol')
        self.assertEqual(sum(r['cost_usd'] for r in records), 1.5)
        truth = read_table(replay.truth, replay.problem).rows
        self.assertEqual(self.evaluator.count('run'), 10)
        journal = self.evaluator.db.execute('SELECT x, y, label FROM evaluations ORDER BY idx').fetchall()
        for row, (x, y, label) in zip(truth, journal):
            self.assertEqual([float(row['x0']), float(row['x1'])], eval(x))
            self.assertEqual(float(row['y']), y)
            self.assertEqual(label, f'block {row["block"]} {row["optimizer"]}')
        self.assertEqual((replay.private / 'campaign/decisions.md').read_text().count('## Block'), 3)
        self.assertTrue((replay.private / 'blocks/002/propose.py').exists())
        self.assertIn('block,attempts,status', (replay.private / 'blocks.csv').read_text())
        # The workspace problem is anonymous and minimizes y.
        problem = load_problem(public)
        self.assertEqual((problem.names, problem.objective, problem.direction), (['x0', 'x1'], 'y', 'minimize'))

    def test_pool_rows_are_evaluated_by_candidate(self):
        replay, public = self.replay(Pool(), budget=6)
        records = replay.run()
        self.assertEqual([r['rows'] for r in records], [4, 2])
        rows = read_table(replay.truth, replay.problem).rows
        self.assertEqual(sorted(r['candidate_id'] for r in rows), ['0', '1', '2', '3', '4', '5'])
        for row in rows:
            self.assertEqual(float(row['y']), float(row['x0']) + float(row['x1']))

    def test_a_block_with_no_proposals_gets_one_retry(self):
        replay, public = self.replay(Sphere(), budget=4)
        working = scripted(public)
        calls = []

        def advise(block, attempt, logs):
            calls.append(attempt)
            return working(block, attempt, logs) if attempt else {}
        replay.advise = advise
        self.assertEqual([(r['attempts'], r['status']) for r in replay.run()], [(2, 'evaluated')])
        replay, _ = self.replay(Sphere(), budget=4, advise=lambda *_: {})
        self.assertEqual([(r['attempts'], r['status']) for r in replay.run()], [(2, 'no proposals')])

    def test_tampering_is_refused(self):
        def edit(path, old, new, count=1):
            text = path.read_text()
            self.assertIn(old, text)
            path.write_text(text.replace(old, new, count))

        def changed_result(public):
            def tamper(block, attempt):
                if block == 2:
                    rows = (public / 'data.csv').read_text().splitlines()
                    rows[1] = rows[1].rsplit(',', 2)[0] + ',0.0,'
                    (public / 'data.csv').write_text('\n'.join(rows) + '\n')
            return tamper

        def rewritten_decisions(public):
            return lambda block, attempt: block == 2 and edit(public / 'decisions.md', '## Block 1', '## Block 1 (revised)')

        def moved_point(public):
            def tamper(block, attempt):
                rows = (public / 'data.csv').read_text().splitlines()
                cells = rows[-1].split(',')
                cells[5] = '0.0'
                rows[-1] = ','.join(cells)
                (public / 'data.csv').write_text('\n'.join(rows) + '\n')
            return tamper

        def manual_row(public):
            def tamper(block, attempt):
                path = public / 'data.csv'
                path.write_text(path.read_text() + '99,1,done,manual,manual,0,0,0,\n')
            return tamper

        cases = {changed_result: 'already recorded', rewritten_decisions: 'append-only',
                 moved_point: 'not what propose.py proposes', manual_row: 'not a pending advisor row'}
        for make, message in cases.items():
            with self.subTest(make.__name__):
                replay, public = self.replay(Sphere())
                replay.advise = scripted(public, make(public))
                with self.assertRaisesRegex(ReplayError, message):
                    replay.run()
                self.assertEqual((replay.records[-1]['status'], replay.records[-1]['cost_usd']), ('refused', .5))
                self.assertIn('refused', (replay.private / 'blocks.csv').read_text())

    def test_links_are_not_followed(self):
        replay, public = self.replay(Sphere())

        def advise(block, attempt, logs):
            (public / 'data.csv').unlink()
            (public / 'data.csv').symlink_to(replay.private / 'campaign/problem.toml')
        replay.advise = advise
        with self.assertRaisesRegex(ReplayError, 'is a link'):
            replay.run()



class Headless(unittest.TestCase):
    def test_usage_from_claude_and_codex_events(self):
        claude = [{'type': 'assistant'}, {'type': 'result', 'num_turns': 12, 'total_cost_usd': .27,
                   'usage': {'input_tokens': 18, 'cache_creation_input_tokens': 20488,
                             'cache_read_input_tokens': 249269, 'output_tokens': 2790}}]
        self.assertEqual({k: v for k, v in usage_of(claude).items() if v},
                         {'turns': 12, 'cost_usd': .27, 'input_tokens': 18, 'cache_write_tokens': 20488,
                          'cache_read_tokens': 249269, 'output_tokens': 2790})
        codex = [{'type': 'turn.completed', 'usage': {'input_tokens': 1000, 'cached_input_tokens': 600,
                                                      'output_tokens': 50}}] * 2
        self.assertEqual({k: v for k, v in usage_of(codex).items() if v},
                         {'turns': 2, 'input_tokens': 800, 'cache_read_tokens': 1200, 'output_tokens': 100})

    def test_instructions_carry_the_policy_and_the_procedure(self):
        text = system_prompt('claude')
        self.assertIn('Optimization Advisor', text)
        self.assertIn('one block, headless', text)
        self.assertIn('isolated MCP tools', text)

    def test_problem_notes_follow_the_task(self):
        noisy = type('Noisy', (Sphere,), {'noisy': True})()
        self.assertIn('noise-free value', problem_toml(noisy, 10, 4, 0))
        self.assertIn('noise = "none"', problem_toml(Sphere(), 10, 4, 0))
        pool = problem_toml(Pool(), 6, 4, 0)
        self.assertIn('[pool]', pool)
        self.assertNotIn('low =', pool)


if __name__ == '__main__':
    unittest.main()
