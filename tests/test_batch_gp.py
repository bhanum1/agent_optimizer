"""GP fit diagnostics and batch qLogNEI, on small synthetic problems."""
import unittest

import numpy as np

from agentoptimizer import batch
from agentoptimizer.batch.gp import fit_gp


def loss(X):
    """Bowl at (0.3, 0.3); the third coordinate does not matter."""
    X = np.atleast_2d(X)
    return ((X[:, :2] - .3) ** 2).sum(1)


RNG = np.random.default_rng(0)
X = RNG.random((20, 3))
Y = loss(X)


class Fit(unittest.TestCase):
    def test_diagnostics(self):
        _, info = fit_gp(X, Y)
        self.assertEqual(info['n'], 20)
        self.assertGreater(info['loo_r2'], .9)
        self.assertGreater(info['lengthscales'][2], 2 * max(info['lengthscales'][:2]))
        self.assertLess(info['noise_share'], .1)


class QLogNEI(unittest.TestCase):
    def test_batch_is_good_distinct_and_deterministic(self):
        p = batch.qlognei(X, Y, 4, seed=1)
        self.assertEqual((p.method, p.X.shape), ('qlognei', (4, 3)))
        np.testing.assert_array_equal(p.X, batch.qlognei(X, Y, 4, seed=1).X)
        self.assertEqual(len({tuple(np.round(row, 6)) for row in p.X}), 4)
        self.assertLess(loss(p.X).mean(), loss(batch.random(X, Y, 4, seed=1).X).mean())
        self.assertIn('lengthscales', p.info)
        self.assertIn('botorch', p.settings['model'])

    def test_bounds_and_pending(self):
        box = np.array([[.5, .5, .5], [.8, .8, .8]])
        p = batch.qlognei(X, Y, 3, bounds=box, seed=0)
        self.assertTrue(((p.X >= .5 - 1e-9) & (p.X <= .8 + 1e-9)).all())
        pending = np.array([[.3, .3, .5]])
        p = batch.qlognei(X, Y, 3, pending=pending, seed=0)
        self.assertGreater(np.linalg.norm(p.X - pending, axis=1).min(), 1e-3)

    def test_pool(self):
        candidates = RNG.random((200, 3))
        p = batch.qlognei(X, Y, 5, candidates=candidates, seed=0)
        self.assertEqual(len(set(p.indices.tolist())), 5)
        np.testing.assert_array_equal(p.X, candidates[p.indices])
        self.assertLess(loss(p.X).mean(), np.median(loss(candidates)))
        with self.assertRaisesRegex(ValueError, 'inside the bounds'):
            batch.qlognei(X, Y, 5, candidates=candidates, bounds=[[0, 0, 0], [.05, .05, .05]])

    def test_failures_and_errors(self):
        failed = np.array([[.3, .3, .9]])
        p = batch.qlognei(X, Y, 2, failed=failed, failed_as='worst', seed=0)
        self.assertEqual(p.info['n'], 21)
        with self.assertRaisesRegex(ValueError, 'at least 2 measured points'):
            batch.qlognei(X[:1], Y[:1], 2)
        with self.assertRaisesRegex(ValueError, "'ignore' or 'worst'"):
            batch.qlognei(X, Y, 2, failed=failed, failed_as='skip')


if __name__ == '__main__':
    unittest.main()
