"""Pool selection and runner integration; each test constructs a fresh pool."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from agentoptimizer.loop import campaign
from agentoptimizer.optimizers.pool_random import PoolRandomSearch
from agentoptimizer.tasks.registry import TASKS
from run_campaign import run_campaign
from run_sweep import _sweep


class OPVPoolIntegrity(unittest.TestCase):
    def test_repeated_query_exhausts_pool_without_replacement(self):
        task = TASKS['opv_pool']()
        ids, values = [], []
        for _ in range(task.n_candidates):
            values.append(task.evaluate(np.full(task.dim, .5)).y)
            ids.append(task.last_candidate)
        self.assertEqual(len(set(ids)), task.n_candidates)
        self.assertEqual(min(values), task.f_opt)
        with self.assertRaisesRegex(ValueError, 'all .* candidates'):
            task.evaluate(np.full(task.dim, .5))
        # A different campaign owns a fresh tested set.
        fresh = TASKS['opv_pool']()
        fresh.evaluate(np.full(fresh.dim, .5))
        self.assertEqual(fresh.last_candidate, ids[0])

    def test_pool_random_selects_each_intended_device(self):
        task = TASKS['opv_pool']()
        optimizer = PoolRandomSearch(task.bounds, seed=0, candidates=task.candidates)
        for point in optimizer.ask(task.n_candidates):
            row = np.flatnonzero(np.all(task.candidates == point, axis=1))
            self.assertEqual(len(row), 1)
            task.evaluate(point)
            self.assertEqual(task.last_candidate, int(task._ids[row[0]]))

    def test_loop_repeats_map_to_new_devices_and_candidates_stay_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = campaign.Campaign.prepare(TASKS['opv_pool'](), 2, 0, Path(tmp) / 'run')
            space = json.loads((run.workspace / 'space.json').read_text())
            self.assertIsNone(space['candidates'])
            replies = [{'script': 'lib/fixed_points.py', 'label': 'probe', 'n_points': 2,
                        'state': 'restart', 'args': {'points': [[.5] * run.task.dim] * 2}}]

            def agent(number, prompt):
                (run.workspace / 'agent/proposal.json').write_text(json.dumps(replies.pop(0)))
                (run.workspace / 'agent/decision.md').write_text('Two probes.')
                return 0
            self.assertEqual(run.run(agent), 'budget')
            self.assertEqual(int(run.task._tested.sum()), 2)  # the repeated point used a second device
            ys = [float(r['y']) for r in run.store.rows]
            self.assertEqual(len(ys), 2)
            public = ''.join((run.workspace / name).read_text()
                             for name in ('space.json', 'PROBLEM.md', 'data.csv', 'decisions.md'))
            for private_key in ('f_opt', 'last_candidate', 'candidate_id'):
                self.assertNotIn(private_key, public)

    def test_oversized_budget_rejected_before_sweep_jobs(self):
        limit = TASKS['opv_pool']().n_candidates
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'budget .* exceeds'):
                run_campaign('opv_pool', 'random_search', 0, limit + 1, tmp)
            with patch('run_sweep._timed_campaign') as launch:
                with self.assertRaisesRegex(ValueError, 'budget .* exceeds'):
                    _sweep('opv_pool', limit + 1, 10, tmp, workers=1)
                launch.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
