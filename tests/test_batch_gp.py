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


class TrustRegion(unittest.TestCase):
    def test_box_around_the_incumbent(self):
        p = batch.trust_region(X, Y, 4, length=.4, seed=0)
        box = np.array(p.info['box'])
        self.assertEqual(p.method, 'trust_region')
        np.testing.assert_allclose(p.info['center'], X[np.argmin(Y)])
        self.assertTrue(((p.X >= box[0] - 1e-9) & (p.X <= box[1] + 1e-9)).all())
        # The third coordinate does not matter, so the box is widest along it.
        width = box[1] - box[0]
        self.assertGreater(width[2], max(width[:2]))
        near = batch.trust_region(X, Y, 4, length=.1, seed=0)
        incumbent = X[np.argmin(Y)]
        self.assertLess(np.linalg.norm(near.X[:, :2] - incumbent[:2], axis=1).mean(),
                        np.linalg.norm(p.X[:, :2] - incumbent[:2], axis=1).mean())

    def test_center_pool_and_errors(self):
        p = batch.trust_region(X, Y, 2, center=[.5, .5, .5], length=.2, seed=0)
        self.assertEqual(p.settings['center'], [.5, .5, .5])
        candidates = RNG.random((30, 3))
        p = batch.trust_region(X, Y, 4, candidates=candidates, length=.05, seed=0)
        self.assertGreater(p.info['length_used'], .05)
        box = np.array(p.info['box'])
        self.assertTrue(((p.X >= box[0] - 1e-9) & (p.X <= box[1] + 1e-9)).all())
        self.assertEqual(len(set(p.indices.tolist())), 4)
        with self.assertRaisesRegex(ValueError, 'length must be'):
            batch.trust_region(X, Y, 2, length=0)
        with self.assertRaisesRegex(ValueError, 'center must be'):
            batch.trust_region(X, Y, 2, center=[2, 0, 0])


if __name__ == '__main__':
    unittest.main()
