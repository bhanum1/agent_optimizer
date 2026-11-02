"""CMA-ES and DE generations warm-started from the data."""
import unittest

import numpy as np

from agentoptimizer import batch

RNG = np.random.default_rng(0)
X = RNG.random((24, 3))
Y = ((X[:, :2] - .3) ** 2).sum(1)


def loss(P):
    return ((P[:, :2] - .3) ** 2).sum(1)


class CMAES(unittest.TestCase):
    def test_generation_starts_at_the_elite(self):
        p = batch.cma_es(X, Y, 8, seed=0)
        self.assertEqual((p.method, p.X.shape), ('cma_es', (8, 3)))
        np.testing.assert_array_equal(p.X, batch.cma_es(X, Y, 8, seed=0).X)
        self.assertTrue(((p.X >= 0) & (p.X <= 1)).all())
        elite = X[np.argsort(Y)[:4]]
        self.assertLess(np.linalg.norm(p.X.mean(0)[:2] - elite.mean(0)[:2]), .25)
        self.assertLess(loss(p.X).mean(), loss(batch.random(X, Y, 8, seed=0).X).mean())
        self.assertEqual(len(p.info['stds']), 3)

    def test_sigma_mu_and_errors(self):
        tight = batch.cma_es(X, Y, 8, sigma=.02, seed=0)
        wide = batch.cma_es(X, Y, 8, sigma=.4, seed=0)
        self.assertLess(tight.X.std(0).mean(), wide.X.std(0).mean())
        self.assertEqual(batch.cma_es(X, Y, 1, seed=0).X.shape, (1, 3))
        with self.assertRaisesRegex(ValueError, 'mu must be'):
            batch.cma_es(X, Y, 4, mu=1)
        with self.assertRaisesRegex(ValueError, 'at least 3 measured'):
            batch.cma_es(X[:2], Y[:2], 4, mu=3)


class DE(unittest.TestCase):
    def test_generation(self):
        p = batch.de(X, Y, 6, seed=2)
        self.assertEqual((p.method, p.X.shape, p.settings['population']), ('de', (6, 3), 6))
        np.testing.assert_array_equal(p.X, batch.de(X, Y, 6, seed=2).X)
        self.assertTrue(((p.X >= 0) & (p.X <= 1)).all())
        best = batch.de(X, Y, 6, seed=2, strategy='best1')
        self.assertLess(loss(best.X).mean(), loss(batch.random(X, Y, 6, seed=2).X).mean())

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, 'at least 4 measured'):
            batch.de(X[:3], Y[:3], 2)
        with self.assertRaisesRegex(ValueError, "'rand1' or 'best1'"):
            batch.de(X, Y, 2, strategy='best2')


class Pools(unittest.TestCase):
    def test_snapped_to_distinct_rows(self):
        candidates = RNG.random((50, 3))
        for proposer in (batch.cma_es, batch.de):
            p = proposer(X, Y, 6, candidates=candidates, seed=0)
            self.assertEqual(len(set(p.indices.tolist())), 6)
            np.testing.assert_array_equal(p.X, candidates[p.indices])


if __name__ == '__main__':
    unittest.main()
