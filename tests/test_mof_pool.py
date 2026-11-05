"""Exact-pool integrity uses synthetic labels, duplicate features and tied optima."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agentoptimizer.loop import campaign, results
from agentoptimizer.tasks.csv_pool import CSVPoolTask
from scripts.run_campaign import optimizers_for_task, run_campaign


class MOFPoolIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.csv = self.root / 'private.csv'
        self.csv.write_text('id,x0,x1,y\nSECRET_A,0,7,1\nSECRET_B,0,7,9\n'
                            'SECRET_C,1,7,9\nSECRET_D,2,7,4\n')

    def campaign(self, budget=4):
        run = campaign.Campaign.prepare(CSVPoolTask(self.csv, 'mof_pool', maximize=True), budget, 2, self.root / 'run')
        return run

    def fixed(self, run, indices, n):
        proposal = {'script': 'lib/fixed_points.py', 'label': 'rows', 'args': {'points': indices},
                    'n_points': n, 'state': 'restart'}
        def agent(number, prompt):
            (run.workspace / 'agent/proposal.json').write_text(json.dumps(proposal))
            (run.workspace / 'agent/decision.md').write_text('Select rows.')
            return 0
        return run.run(agent, max_calls=1)

    def test_exact_identity_and_tied_rank_conventions(self):
        task = CSVPoolTask(self.csv, 'mof_pool', maximize=True)
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

    def test_loop_rejects_duplicate_rows_and_keeps_labels_private(self):
        run = self.campaign()
        self.fixed(run, [1, 2, 1, 0], 4)
        decisions = (run.workspace / 'decisions.md').read_text()
        self.assertIn('candidate 1 was already evaluated', decisions)
        self.assertEqual([r['candidate_index'] for r in run.store.rows], ['1', '2'])
        self.assertEqual([float(r['y']) for r in run.store.rows], [-9., -9.])
        report = results.audit(run, None)
        results.write(run, {'task': 'mof', 'agent': 'fake', 'model': '-', 'effort': '-'}, 'agent', report, '-')
        meta = json.loads((run.root / 'results/meta.json').read_text())
        self.assertEqual(meta['best_candidate_index'], 1)  # Keep first observed tie.
        self.assertEqual(meta['seed'], 2)
        # The task-specific public files (SKILL.md and lib/ are the same for every task).
        public = ''.join((run.workspace / name).read_text()
                         for name in ('space.json', 'PROBLEM.md', 'data.csv', 'decisions.md'))
        for hidden in ('SECRET_', 'f_opt', 'raw_y', 'rank', 'private.csv'):
            self.assertNotIn(hidden, public)

    def test_baseline_dispatch_preserves_exact_rows_and_minimization(self):
        factory = lambda: CSVPoolTask(self.csv, 'mof_pool', maximize=True)
        self.assertEqual(set(optimizers_for_task(factory())), {'gp_bo', 'sobol_gp_bo', 'turbo', 'pool_random'})
        with patch.dict('scripts.run_campaign.TASKS', {'synthetic': factory}):
            out = run_campaign('synthetic', 'pool_random', 0, 4, self.root / 'results')
            with (out / 'trace.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len({r['candidate_index'] for r in rows}), 4)
            self.assertEqual(float(rows[-1]['y_best_so_far']), -9.)


if __name__ == '__main__':
    unittest.main()
