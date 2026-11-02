"""The Campaign API for proposal scripts, and three blocks end to end with no model."""
import csv
import math
from pathlib import Path
import tempfile
import textwrap
import unittest

import numpy as np

from advisor_fixtures import campaign, quiet
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


# Three blocks a model might have written: a space-filling start, GP-BO from
# all the data, and a local batch around the incumbent.
SCRIPTS = {
    1: '''
        from agentoptimizer.advisor import Campaign
        from agentoptimizer.optimizers.sobol_search import SobolSearch
        campaign = Campaign.load(__file__)
        campaign.write_proposals(SobolSearch(campaign.unit_bounds, seed=1).ask(8), label='sobol')
    ''',
    2: '''
        from agentoptimizer.advisor import Campaign
        from agentoptimizer.optimizers.gp_bo import GPBO
        campaign = Campaign.load(__file__)
        X, y = campaign.observations()
        gp = GPBO(campaign.unit_bounds, seed=2)
        gp.tell(X, y)
        campaign.write_proposals(gp.ask(1), label='gp_bo')
    ''',
    3: '''
        import numpy as np
        from agentoptimizer.advisor import Campaign
        campaign = Campaign.load(__file__)
        x, _ = campaign.incumbent()
        rng = np.random.default_rng(3)
        U = np.clip(x + rng.normal(0, 0.05, size=(4, campaign.dim)), 0, 1)
        campaign.write_proposals(U, label='local_gaussian')
    ''',
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

    def test_write_proposals_rounds_and_clips(self):
        (self.folder / 'blocks/002').mkdir(parents=True)
        c = Campaign.load(self.folder / 'blocks/002')
        stack, _ = quiet()
        with stack:
            path = c.write_proposals([[0.5000000001, 0.5, 0.0], [1.2, -0.1, 0.99]], label='test')
            c.write_proposals([[140.000000000003, 0.05, 7]], label='raw', raw=True)
        with path.open() as f:
            self.assertEqual(list(csv.reader(f))[1:], [['140.0', '0.05', '7', 'raw', '']])
        stack, _ = quiet()
        with stack:
            c.write_proposals([[0.5000000001, 0.5, 0.0], [1.2, -0.1, 0.99]], label='test')
        with path.open() as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ['temperature', 'conc', 'stir', 'optimizer', 'notes'])
        self.assertEqual(rows[1][:3], ['140.0', '0.0316228', '1'])
        self.assertEqual(rows[2][:3], ['200.0', '0.001', '30'])
        with self.assertRaisesRegex(ProblemError, 'inside blocks/NNN'):
            Campaign.load(self.folder).write_proposals([[0, 0, 0]], label='x')
        with self.assertRaisesRegex(ValueError, '3 coordinates'):
            c.write_proposals([[0, 0]], label='x')


class Loop(unittest.TestCase):
    def test_three_blocks_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = campaign(Path(tmp) / 'c')
            for number, script in SCRIPTS.items():
                opened, block = new_block(folder)
                self.assertEqual(opened, number)
                (block / 'propose.py').write_text(textwrap.dedent(script))
                with (folder / 'decisions.md').open('a') as f:
                    f.write(f'## Block {number}\nSee propose.py.\n\n')
                rows, issues = run_block(folder)
                self.assertEqual([i for i in issues if i.level == 'error'], [])
                first = (block / 'proposals.csv').read_text()
                ids = accept_block(folder)
                self.assertEqual(len(ids), len(rows))

                # The script reproduces its proposals exactly when rerun.
                run_block(folder, number)
                self.assertEqual((block / 'proposals.csv').read_text(), first)

                # The "lab" measures the pending rows.
                problem, table, data = load_campaign(folder)
                results = Path(tmp) / f'results_{number}.csv'
                with results.open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id', 'yield'])
                    for i in np.flatnonzero(data.where('pending')):
                        writer.writerow([data.ids[i], f'{yield_of(*data.X[i]):.6f}'])
                record_results(folder, results)

            problem, _, data = load_campaign(folder)
            summary = summarize(problem, data)
            self.assertEqual(summary['counts'], {'done': 13, 'pending': 0, 'failed': 0})
            self.assertEqual([b['optimizers'] for b in summary['blocks']],
                             [['sobol'], ['gp_bo'], ['local_gaussian']])
            self.assertEqual(summary['budget']['remaining'], 7)
            best = summary['incumbent']['value']
            self.assertGreaterEqual(best, max(b['best'] for b in summary['blocks']))
            self.assertGreater(best, .5)


if __name__ == '__main__':
    unittest.main()
