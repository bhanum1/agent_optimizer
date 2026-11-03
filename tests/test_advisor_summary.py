"""The data summary, checked against values worked out by hand."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from advisor_fixtures import PROBLEM, campaign, quiet
from agentoptimizer.advisor.cli import main
from agentoptimizer.advisor.data import load_campaign
from agentoptimizer.advisor.summary import format_summary, summarize

# Unit coordinates: id 1 is (0, 0, 1/60); ids 2 and 3 are (0.5, 0.5, 14.5/30);
# id 4 is (1, 1, 29.5/30). Ids 2 and 3 are replicates.
ROWS = '''\
    1,0,done,import,import,80,0.001,1,0.40,
    2,1,done,advisor,sobol,140,0.0316227766016838,15,0.50,
    3,1,done,advisor,sobol,140,0.0316227766016838,15,0.54,
    4,1,failed,advisor,sobol,200,1.0,30,,
    5,2,done,advisor,gp_bo,150,0.05,16,0.48,
    6,2,pending,advisor,gp_bo,160,0.06,17,,
'''


class Summary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def summary(self, rows=ROWS, problem=PROBLEM):
        campaign(self.folder, rows, problem=problem)
        problem, _, data = load_campaign(self.folder)
        return summarize(problem, data)

    def test_hand_worked_campaign(self):
        s = self.summary()
        self.assertEqual(s['counts'], {'done': 4, 'pending': 1, 'failed': 1})
        self.assertEqual(s['budget'], {'total': 20, 'used': 5, 'pending': 1, 'remaining': 14})
        self.assertEqual((s['incumbent']['id'], s['incumbent']['value'], s['incumbent']['at_bounds']), (3, .54, []))
        self.assertEqual([r['id'] for r in s['top']], [3, 2, 5, 1])

        first, second, third = s['blocks']
        self.assertEqual((first['best'], first['improved_by'], first['spread']), (.40, None, None))
        self.assertEqual((second['optimizers'], second['done'], second['failed']), (['sobol'], 2, 1))
        self.assertAlmostEqual(second['improved_by'], .14)
        # Median distance from ids 2-4 to id 1, and median pairwise distance within ids 2-4.
        near = math.sqrt(.5 + (14.5 / 30 - 1 / 60) ** 2)
        self.assertAlmostEqual(second['distance_to_previous_incumbent'], near / math.sqrt(3))
        self.assertAlmostEqual(second['spread'], math.sqrt(.75) / math.sqrt(3))
        self.assertEqual((third['best'], third['best_so_far'], third['improved_by'], third['pending']),
                         (.48, .54, 0., 1))
        self.assertEqual(s['blocks_since_improvement'], 1)

        self.assertEqual(s['replicates']['groups'], 1)
        self.assertAlmostEqual(s['replicates']['pooled_sd'], math.sqrt(.0008))
        self.assertEqual(s['near_duplicate_pairs'], 1)
        self.assertEqual(s['explored']['temperature'], {'low': 80., 'high': 200., 'unit_span': 1.})
        self.assertEqual(s['failures'], {'count': 1, 'rate': .2, 'by_block': {'1': 1}})
        json.dumps(s, allow_nan=False)

    def test_bounds_direction_and_empty(self):
        s = self.summary(ROWS.splitlines()[0] + '\n')
        self.assertEqual(s['incumbent']['at_bounds'], [{'parameter': name, 'side': 'low'}
                                                       for name in ('temperature', 'conc', 'stir')])
        s = self.summary(problem=PROBLEM.replace('maximize', 'minimize'))
        self.assertEqual(s['incumbent']['id'], 1)
        self.assertEqual(s['blocks_since_improvement'], 2)
        (self.folder / 'data.csv').unlink()
        s = self.summary(rows='')
        self.assertEqual((s['incumbent'], s['blocks'], s['budget']['remaining']), (None, [], 20))
        self.assertIn('incumbent: none yet', format_summary(s))

    def test_cli(self):
        campaign(self.folder, ROWS)
        stack, out = quiet()
        with stack:
            self.assertEqual(main(['summarize', str(self.folder)]), 0)
        text = out.getvalue()
        self.assertIn('incumbent: id 3 (block 1), yield 0.54', text)
        self.assertIn('blocks since the incumbent improved: 1', text)
        stack, out = quiet()
        with stack:
            main(['summarize', str(self.folder), '--json'])
        self.assertEqual(json.loads(out.getvalue())['incumbent']['id'], 3)


if __name__ == '__main__':
    unittest.main()
