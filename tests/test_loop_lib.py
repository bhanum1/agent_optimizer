"""Library optimizers: each runs four pickled blocks on a shifted quadratic.
Run with the agent runtime (.agent-venv), which has pybobyqa and botorch."""

import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"  # as in the sandbox; threaded BLAS makes BOBYQA ~1000x slower

from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

LOOP = Path(__file__).resolve().parents[1] / "src" / "agentoptimizer" / "loop"
sys.path.insert(0, str(LOOP))
import host  # noqa: E402

SHIFT = np.array([1.3, -2.0, 0.5, 3.1])
BOX = {"dim": 4, "bounds": [[-5, 5]] * 4, "n_constraints": 0, "noisy": False,
       "indexed_pool": False, "candidates": None}


def f(x):
    d = np.asarray(x) - SHIFT
    return float(d @ d + 0.5 * d[0] * d[1])


def drive(name, space, objective, blocks=4, per_block=30, args=None):
    """Run `blocks` blocks, pickling between them as the runner does; return the values."""
    opt = host.load_class(LOOP / "lib" / f"{name}.py")(space, **(args or {"seed": 0}))
    ys = []
    for _ in range(blocks):
        for _ in range(per_block):
            X = opt.ask(1)
            if not X:
                break
            y, g = objective(X[0])
            ys.append(y)
            opt.tell(X, [y], [g])
        opt = pickle.loads(pickle.dumps(opt))
    return ys


class LibraryTests(unittest.TestCase):
    def test_methods_make_progress(self):
        thresholds = {"cma_es": 0.1, "nelder_mead": 0.01, "pso": 1.0, "lbfgsb": 1e-4, "powell": 1e-4,
                      "bobyqa": 1e-4, "cobyla": 1e-3, "random_search": 20, "sobol": 20, "de": 20}
        for name, limit in thresholds.items():
            with self.subTest(name):
                self.assertLess(min(drive(name, BOX, lambda x: (f(x), []))), limit)

    def test_model_based_methods_make_progress(self):
        for name in ("gp_bo", "turbo"):
            with self.subTest(name):
                self.assertLess(min(drive(name, BOX, lambda x: (f(x), []), blocks=3, per_block=20)), 1.0)

    def test_cobyla_respects_constraints(self):
        space = dict(BOX, n_constraints=1)
        # Unconstrained optimum is SHIFT; require x0 <= 0, so the optimum moves to the boundary.
        opt =host.load_class(LOOP / "lib" / "cobyla.py")(space, seed=0)
        best = None
        for _ in range(150):
            X = opt.ask(1)
            if not X:
                break
            x = X[0]
            opt.tell(X, [f(x)], [[x[0]]])
            if x[0] <= 1e-6 and (best is None or f(x) < best[0]):
                best = (f(x), x)
        self.assertIsNotNone(best)
        self.assertLess(abs(best[1][0]), 1e-3)  # the constraint is active at the optimum

    def test_pool_methods_pick_unused_candidates(self):
        rng = np.random.default_rng(0)
        cands = rng.uniform(-5, 5, (60, 4))
        space = dict(BOX, indexed_pool=True, candidates=cands.tolist())
        for name in ("random_search", "gp_bo", "turbo", "cma_es"):
            with self.subTest(name):
                opt = host.load_class(LOOP / "lib" / f"{name}.py")(space, seed=0)
                picked = []
                for _ in range(30):
                    (i,) = opt.ask(1)
                    picked.append(i)
                    opt.tell([i], [f(cands[i])], [[]])
                    opt = pickle.loads(pickle.dumps(opt))
                self.assertEqual(len(set(picked)), 30)

    def test_continue_resumes_nelder_mead_exactly(self):
        straight = drive("nelder_mead", BOX, lambda x: (f(x), []), blocks=1, per_block=60)
        pickled = drive("nelder_mead", BOX, lambda x: (f(x), []), blocks=6, per_block=10)
        self.assertEqual(straight, pickled)

    def test_continue_resumes_callback_solvers_exactly(self):
        # A run that converges ends its block; the later blocks then start a new run, so compare that prefix.
        for name in ("bobyqa", "lbfgsb", "powell", "cobyla"):
            with self.subTest(name):
                straight = drive(name, BOX, lambda x: (f(x), []), blocks=1, per_block=60)
                pickled = drive(name, BOX, lambda x: (f(x), []), blocks=6, per_block=10)
                self.assertGreater(len(straight), 20)
                self.assertEqual(straight, pickled[:len(straight)])

    def test_fixed_coordinates_hold_and_the_rest_converge(self):
        space, asked = BOX, []

        def objective(x):
            asked.append(x)
            return f(x), []
        for name in ("bobyqa", "lbfgsb"):
            with self.subTest(name):
                asked.clear()
                ys = drive(name, space, objective, args={"seed": 0, "fixed": {"0": 1.3, "2": 0.5}})
                self.assertTrue(all(x[0] == 1.3 and x[2] == 0.5 for x in asked))
                self.assertLess(min(ys), 1e-4)

    def test_replay_that_diverges_restarts_from_the_best_point(self):
        opt = host.load_class(LOOP / "lib" / "bobyqa.py")(BOX, seed=0)
        for _ in range(15):
            X = opt.ask(1)
            opt.tell(X, [f(X[0])], [[]])
        state = pickle.loads(pickle.dumps(opt))
        point, value, raw = state.calls[0]
        state.calls[0] = (point + 0.1, value, raw)
        (x,) = state.ask(1)  # replay fails at call 0; a fresh run starts at the best point
        self.assertTrue(np.allclose(x, state.from_unit(state.U[np.argmin(state.y)])))


if __name__ == "__main__":
    unittest.main()
