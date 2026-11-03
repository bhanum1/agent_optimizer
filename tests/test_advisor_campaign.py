"""The Campaign API for proposal scripts."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from advisor_fixtures import campaign, quiet
from agentoptimizer import batch
from agentoptimizer.advisor import Campaign, ProblemError

ROWS = '''\
    1,0,done,import,import,80,0.001,1,0.40,
    2,1,done,advisor,sobol,140,0.1,15,0.54,
    3,1,pending,advisor,sobol,200,1.0,30,,
'''


class CampaignApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = campaign(Path(self.tmp.name) / 'c', ROWS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_views(self):
        (self.folder / 'blocks/002').mkdir(parents=True)
        c = Campaign.load(self.folder / 'blocks/002/propose.py')
        self.assertEqual((c.root, c.block, c.dim, c.remaining), (self.folder.resolve(), 2, 3, 17))
        X, y = c.observations()
        np.testing.assert_allclose(y, [-.40, -.54])  # maximize: loss is the negated yield
        np.testing.assert_allclose(X[0], [0, 0, 1 / 60])
        Xr, yr = c.observations(raw=True)
        np.testing.assert_allclose(Xr[1], [140, .1, 15])
        np.testing.assert_allclose(yr, [.40, .54])
        np.testing.assert_allclose(c.pending(raw=True), [[200, 1., 30]])
        self.assertEqual(c.incumbent(raw=True)[1], .54)
        self.assertEqual(c.incumbent()[1], -.54)
        self.assertIsNone(Campaign.load(self.folder).block)

    def test_write_rounds_and_clips(self):
        (self.folder / 'blocks/002').mkdir(parents=True)
        c = Campaign.load(self.folder / 'blocks/002')
        self.assertEqual(c.round, 1)
        stack, _ = quiet()
        with stack:
            path = c.write(batch.Proposal('test', np.array([[0.5000000001, 0.5, 0.0], [1., 0., 0.99]])))
            with self.assertRaisesRegex(ProblemError, 'inside blocks/NNN'):
                Campaign.load(self.folder).write(batch.Proposal('x', np.zeros((1, 3))))
            with self.assertRaisesRegex(ValueError, '3 coordinates'):
                c.write(batch.Proposal('x', np.zeros((1, 2))))
        with path.open() as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ['temperature', 'conc', 'stir', 'optimizer', 'notes'])
        self.assertEqual(rows[1][:4], ['140.0', '0.0316228', '1', 'test'])
        self.assertEqual(rows[2][:3], ['200.0', '0.001', '30'])


if __name__ == '__main__':
    unittest.main()
