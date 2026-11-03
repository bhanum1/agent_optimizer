"""Exact-pool integrity uses synthetic labels, duplicate features and tied optima."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agentoptimizer.blind_runner.evaluator import Evaluator
from agentoptimizer.tasks.mof_pool import MOFPoolTask
from scripts.run_campaign import optimizers_for_task, run_campaign


class MOFPoolIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / 'private.csv'
        self.csv.write_text('id,x0,x1,y\nSECRET_A,0,7,1\nSECRET_B,0,7,9\n'
                            'SECRET_C,1,7,9\nSECRET_D,2,7,4\n')

    def evaluator(self, budget=4):
        e = Evaluator({'budget': budget, 'seed': 2, 'campaigns': [{'id': 'run'}]},
                      self.root / 'journal.sqlite', lambda _: MOFPoolTask(self.csv))
        return e

    def call(self, e, op, **kwargs):
        return e.handle('run', {'request_id': uuid.uuid4().hex, 'op': op, **kwargs})

    def initialize(self, e):
        self.assertTrue(self.call(e, 'label', name='test', documents={'decisions.md': '## Block 1'})['ok'])

    def test_exact_identity_and_tied_rank_conventions(self):
        task = MOFPoolTask(self.csv)
        self.assertEqual(task.evaluate_index(1).y, -9.)
        self.assertEqual(task.evaluate_index(0).y, -1.)
        with self.assertRaisesRegex(ValueError, 'already'):
            task.evaluate_index(1)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            task.evaluate([.5, 7.])
        self.assertEqual(task.offline_observation(1)['descending_rank'], 1)
        self.assertEqual(task.offline_observation(2)['descending_rank'], 1)
        self.assertEqual(task.offline_observation(1)['rank'], 3)
        self.assertEqual(task.f_opt, -9.)

    def test_retries_duplicates_invalid_rows_and_restart(self):
        e = self.evaluator()
        self.initialize(e)
        request = {'request_id': uuid.uuid4().hex, 'op': 'evaluate', 'candidate_index': 1}
        first = e.handle('run', request)
        self.assertEqual(first['result'], -9.)
        self.assertEqual(e.handle('run', request), first)
        for index in (1, True, -1, 4, 1.5, '2'):
            self.assertFalse(self.call(e, 'evaluate', candidate_index=index)['ok'])
        self.assertFalse(self.call(e, 'evaluate', x=[0., 7.])['ok'])
        self.assertEqual(e.count('run'), 1)
        e.db.close()
        e = self.evaluator()
        self.addCleanup(e.db.close)
        self.assertEqual(e.handle('run', request), first)
        self.assertFalse(self.call(e, 'evaluate', candidate_index=1)['ok'])
        self.assertEqual(self.call(e, 'evaluate', candidate_index=0)['result'], -1.)
        self.assertEqual(self.call(e, 'evaluate', candidate_index=2)['result'], -9.)
        self.assertEqual(e.status('run')['seed'], 2)
        public = json.dumps([e.status('run'), e.scored_trace('run')])
        for hidden in ('SECRET_', 'f_opt', 'raw_y', 'rank', 'private.csv'):
            self.assertNotIn(hidden, public)

    def test_baseline_dispatch_preserves_exact_rows_and_minimization(self):
        factory = lambda: MOFPoolTask(self.csv)
        self.assertEqual(set(optimizers_for_task(factory())), {'gp_bo', 'turbo', 'pool_random'})
        with patch.dict('scripts.run_campaign.TASKS', {'synthetic': factory}):
            out = run_campaign('synthetic', 'pool_random', 0, 4, self.root / 'results')
            with (out / 'trace.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len({r['candidate_index'] for r in rows}), 4)
            self.assertEqual(float(rows[-1]['y_best_so_far']), -9.)


if __name__ == '__main__':
    unittest.main()
