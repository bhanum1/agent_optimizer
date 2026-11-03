"""The Campaign API for proposal scripts, and three blocks end to end with no model."""
import csv
import math
from pathlib import Path
import tempfile
import textwrap
import unittest

import numpy as np

from advisor_fixtures import PROBLEM, campaign, quiet
from agentoptimizer import batch
from agentoptimizer.advisor import Campaign, ProblemError
from agentoptimizer.advisor.blocks import accept_block, new_block, run_block
from agentoptimizer.advisor.data import load_campaign
from agentoptimizer.advisor.results import record_results
from agentoptimizer.advisor.summary import summarize

ROWS = '''\
    1,0,done,import,import,80,0.001,1,0.40,
    2,1,done,advisor,sobol,140,0.1,15,0.54,
    3,1,pending,advisor,sobol,200,1.0,30,,
'''


def yield_of(temperature, conc, stir):
    """Synthetic objective to maximize: best at 150 C, 0.03 M, 12 min."""
    return (1 - ((temperature - 150) / 120) ** 2 - (math.log10(conc) + 1.5) ** 2 / 4
            - ((stir - 12) / 30) ** 2)


# Three blocks a model might have written: a space-filling start, batch GP-BO
# from all the data, and local points of its own around the incumbent.
SCRIPTS = {
    1: (8, '''
        from agentoptimizer import batch
        SIZE = 8
        def propose(campaign, q):
            return batch.space_filling(**campaign.inputs(), q=q, seed=campaign.round)
    '''),
    2: (4, '''
        from agentoptimizer import batch
        SIZE = 4
        def propose(campaign, q):
            return batch.qlognei(**campaign.inputs(), q=q, seed=2)
    '''),
    3: (4, '''
        import numpy as np
        from agentoptimizer import batch
        SIZE = 4
        def propose(campaign, q):
            x, _ = campaign.incumbent()
            rng = np.random.default_rng(3)
            return batch.Proposal('local_gaussian', np.clip(x + rng.normal(0, 0.05, size=(q, campaign.dim)), 0, 1))
    '''),
}


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


class Loop(unittest.TestCase):
    def test_three_blocks_in_rounds_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = campaign(Path(tmp) / 'c', problem=PROBLEM.replace('block_size = 4', 'round_size = 4'))
            for number, (size, script) in SCRIPTS.items():
                opened, block = new_block(folder)
                self.assertEqual(opened, number)
                (block / 'propose.py').write_text(textwrap.dedent(script))
                with (folder / 'decisions.md').open('a') as f:
                    f.write(f'## Block {number}\nSee propose.py.\n\n')
                for round_ in range(size // 4):
                    rows, issues = run_block(folder)
                    self.assertEqual(([i for i in issues if i.level == 'error'], len(rows)), ([], 4))
                    accept_block(folder)
                    problem, table, data = load_campaign(folder)
                    self.assertEqual(data.in_block(number), 4 * (round_ + 1))

                    # The "lab" measures the pending rows.
                    results = Path(tmp) / f'results_{number}.csv'
                    with results.open('w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['id', 'yield'])
                        for i in np.flatnonzero(data.where('pending')):
                            writer.writerow([data.ids[i], f'{yield_of(*data.X[i]):.6f}'])
                    record_results(folder, results)

            problem, _, data = load_campaign(folder)
            summary = summarize(problem, data)
            self.assertEqual(summary['counts'], {'done': 16, 'pending': 0, 'failed': 0})
            self.assertEqual([b['optimizers'] for b in summary['blocks']],
                             [['sobol', 'space_filling'], ['qlognei'], ['local_gaussian']])
            self.assertEqual(summary['budget']['remaining'], 4)
            best = summary['incumbent']['value']
            self.assertGreaterEqual(best, max(b['best'] for b in summary['blocks']))
            self.assertGreater(best, .5)


if __name__ == '__main__':
    unittest.main()
