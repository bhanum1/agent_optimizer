"""Batch proposers, shared parts: inputs, snapping, random and space-filling designs."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from advisor_fixtures import campaign, quiet
from agentoptimizer import batch
from agentoptimizer.advisor import Campaign
from agentoptimizer.advisor.data import load_campaign

RNG = np.random.default_rng(0)
X = RNG.random((10, 3))
Y = ((X - .3) ** 2).sum(1)


class Inputs(unittest.TestCase):
    def test_errors(self):
        cases = [
            (dict(X=[], y=[], q=2), 'pass dim'),
            (dict(X=X, y=Y[:-1], q=2), '10 rows but y has 9'),
            (dict(X=X, y=np.r_[Y[:-1], np.nan], q=2), 'y must be finite'),
            (dict(X=X * 2, y=Y, q=2), 'inside the unit cube'),
            (dict(X=X, y=Y, q=0), 'positive integer'),
            (dict(X=X, y=Y, q=4, candidates=X[:3]), 'exceeds the 3 candidates'),
            (dict(X=X, y=Y, q=2, pending=np.ones((1, 2))), r'shape \(n, 3\)'),
        ]
        for kwargs, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                batch.check_inputs(**kwargs)

    def test_snap_is_distinct_and_nearest(self):
        candidates = np.array([[0., 0.], [.1, 0.], [1., 1.]])
        self.assertEqual(batch.snap(np.array([[0., 0.], [0., 0.], [.9, .9]]), candidates).tolist(), [0, 1, 2])


class Space(unittest.TestCase):
    def test_first_design_is_sobol(self):
        p = batch.space_filling([], [], 8, dim=3, seed=1)
        self.assertEqual((p.method, p.X.shape), ('sobol', (8, 3)))
        np.testing.assert_array_equal(p.X, batch.space_filling([], [], 8, dim=3, seed=1).X)
        self.assertFalse(np.array_equal(p.X, batch.space_filling([], [], 8, dim=3, seed=2).X))

    def test_maximin_keeps_away_from_placed_points(self):
        pending = RNG.random((3, 3))
        p = batch.space_filling(X, Y, 6, pending=pending, seed=0)
        r = batch.random(X, Y, 6, seed=0)
        placed = np.vstack([X, pending])
        gap = lambda P: np.min(np.linalg.norm(P[:, None] - placed[None], axis=-1), axis=1)
        self.assertGreater(gap(p.X).min(), gap(r.X).min())
        self.assertAlmostEqual(p.info['min_distance_to_placed'], gap(p.X).min() / np.sqrt(3))
        self.assertEqual(len({tuple(row) for row in p.X}), 6)

    def test_pool_picks_distinct_rows(self):
        candidates = RNG.random((40, 3))
        for proposer in (batch.space_filling, batch.random):
            p = proposer(X, Y, 5, candidates=candidates, seed=3)
            self.assertEqual(len(set(p.indices.tolist())), 5)
            np.testing.assert_array_equal(p.X, candidates[p.indices])

    def test_describe(self):
        text = batch.space_filling(X, Y, 2, seed=0).describe()
        self.assertIn('space_filling: 2 points', text)
        self.assertIn('setting n_candidates = 4096', text)


class Combine(unittest.TestCase):
    def test_box_parts_keep_their_labels(self):
        first = batch.space_filling(X, Y, 3, seed=0)
        second = batch.random(X, Y, 2, pending=first.X, seed=1)
        block = batch.combine(first, second)
        self.assertEqual((block.method, block.X.shape), ('space_filling+random', (5, 3)))
        self.assertEqual(block.row_labels(), ['space_filling'] * 3 + ['random'] * 2)
        self.assertIn('1:space_filling', block.settings)

    def test_pool_parts_do_not_repeat_rows(self):
        candidates = RNG.random((12, 3))
        first = batch.random(X, Y, 4, candidates=candidates, seed=0)
        with self.assertRaisesRegex(ValueError, 'same candidate twice'):
            batch.combine(first, batch.random(X, Y, 4, candidates=candidates, seed=0))
        for proposer in (batch.random, batch.space_filling):
            second = proposer(X, Y, 8, candidates=candidates, pending=first.X, seed=0)
            self.assertEqual(len(set(batch.combine(first, second).indices.tolist())), 12)
        with self.assertRaisesRegex(ValueError, 'exceeds the 8 candidates'):
            batch.random(X, Y, 9, candidates=candidates, pending=first.X)
        with self.assertRaisesRegex(ValueError, 'pool rows with free points'):
            batch.combine(first, batch.random(X, Y, 1, seed=0))


class CampaignHookup(unittest.TestCase):
    def test_box_and_pool_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = campaign(Path(tmp) / 'c', '1,0,done,import,import,100,0.01,5,0.4,\n')
            (folder / 'blocks/001').mkdir(parents=True)
            c = Campaign.load(folder / 'blocks/001')
            stack, out = quiet()
            with stack:
                c.write(batch.space_filling(**c.inputs(), q=3, seed=0))
            self.assertIn('space_filling: 3 points', out.getvalue())
            text = (folder / 'blocks/001/proposals.csv').read_text().splitlines()
            self.assertEqual((len(text), text[1].split(',')[-2]), (4, 'space_filling'))
            inputs = c.inputs()
            first = batch.space_filling(**inputs, q=2, seed=0)
            second = batch.random(**{**inputs, 'pending': np.vstack([inputs['pending'], first.X])}, q=1, seed=0)
            stack, _ = quiet()
            with stack:
                c.write(batch.combine(first, second))
            labels = [line.split(',')[-2] for line in (folder / 'blocks/001/proposals.csv').read_text().splitlines()[1:]]
            self.assertEqual(labels, ['space_filling', 'space_filling', 'random'])

        import test_advisor_pool as pools
        with tempfile.TemporaryDirectory() as tmp:
            folder = pools.make(Path(tmp) / 'p')
            (folder / 'data.csv').write_text('id,block,status,source,optimizer,candidate_id,x1,x2,x3,score,notes\n'
                                             '1,0,done,import,import,C001,0.37037037037037035,1,7,1.0,\n')
            (folder / 'blocks/001').mkdir(parents=True)
            c = Campaign.load(folder / 'blocks/001')
            inputs = c.inputs()
            self.assertEqual(len(inputs['candidates']), 29)
            stack, _ = quiet()
            with stack:
                c.write(batch.space_filling(**inputs, q=4, seed=0))
            rows = (folder / 'blocks/001/proposals.csv').read_text().splitlines()[1:]
            self.assertEqual(len({r.split(',')[0] for r in rows}), 4)
            self.assertNotIn('C001', [r.split(',')[0] for r in rows])


if __name__ == '__main__':
    unittest.main()
