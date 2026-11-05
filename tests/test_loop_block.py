"""Block driver checks with a toy task and toy optimizers (no sandbox, no agent)."""

from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from agentoptimizer.loop import block, data
from agentoptimizer.tasks.base import Evaluation

COUNTER = """
import json
class Optimizer:
    # Proposes x = (k, 0) for k = 1, 2, ...; k survives save/load.
    def __init__(self, space, start=0, crash_at=None):
        self.k, self.crash_at, self.told = start, crash_at, 0
    def ask(self, n):
        out = []
        for _ in range(n):
            self.k += 1
            if self.k == self.crash_at:
                raise RuntimeError("boom")
            out.append([float(self.k), 0.0])
        return out
    def tell(self, X, y, g):
        self.told += len(y)
    def save(self, path):
        open(path, "w").write(json.dumps([self.k, self.told]))
    @classmethod
    def load(cls, path):
        opt = cls(None)
        opt.k, opt.told = json.loads(open(path).read())
        return opt
"""


class Quadratic:
    name, dim, n_constraints, noisy, pool, f_opt = "quad", 2, 0, False, False, 0.0
    bounds = np.array([[-10.0, 10.0]] * 2)

    def evaluate(self, x):
        return Evaluation(float(np.sum(np.square(x))), (), 0.0)


POOL = """
class Optimizer:
    # Proposes candidate indices 0, 1, 2, ... and then repeats 0.
    def __init__(self, space, repeat_after=99):
        self.k, self.repeat_after = 0, repeat_after
    def ask(self, n):
        self.k += 1
        return [0 if self.k > self.repeat_after else self.k - 1]
    def tell(self, X, y, g):
        pass
    def save(self, path):
        pass
"""


class Pool(Quadratic):
    pool, indexed_pool, n_candidates = True, True, 4
    candidates = np.arange(8.0).reshape(4, 2)

    def evaluate_index(self, i):
        return self.evaluate(self.candidates[i])


class LoopBlockTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.task = Quadratic()
        self.store = data.Store(self.task, self.dir / "data.csv", self.dir / "private" / "journal.csv")
        data.write_space(self.task, self.dir / "space.json")

    def script(self, source, name="opt.py"):
        path = self.dir / name
        path.write_text(textwrap.dedent(source))
        return path

    def run_block(self, number, n, state="restart", args=None, script=None, warm=False):
        b = block.Block(number, "counter", script or self.script(COUNTER), args or {}, n, state, warm)
        return block.run(self.task, self.store, b, self.dir / "state.json", self.dir / "space.json")

    def test_state_continues_across_blocks(self):
        self.assertEqual(self.run_block(1, 3), block.Result(3, None, True))
        self.assertEqual(self.run_block(2, 2, state="continue"), block.Result(2, None, True))
        X, y, _ = self.store.history()
        self.assertEqual([x[0] for x in X], [1, 2, 3, 4, 5])
        self.assertEqual(y, [1, 4, 9, 16, 25])
        self.assertEqual((self.dir / "data.csv").read_text().splitlines()[0], "eval,block,optimizer,x0,x1,y")

    def test_crash_keeps_paid_evaluations(self):
        result = self.run_block(1, 5, args={"crash_at": 3})
        self.assertEqual((result.used, result.state_saved), (2, False))
        self.assertIn("boom", result.error)
        self.assertEqual(len(self.store), 2)

    def test_out_of_bounds_proposal_is_rejected(self):
        result = self.run_block(1, 5, args={"start": 9})
        self.assertEqual(result.used, 1)
        self.assertIn("outside the bounds", result.error)

    def test_warm_start_tells_history(self):
        self.run_block(1, 2)
        self.run_block(2, 1, args={"start": 5}, warm=True)
        self.assertEqual(self.dir.joinpath("state.json").read_text(), "[6, 3]")

    def test_pickled_state_loads_in_next_block(self):
        source = "import pickle\n" + COUNTER.replace(
            "    def save(self, path):\n        open(path, \"w\").write(json.dumps([self.k, self.told]))",
            "    def save(self, path):\n        pickle.dump(self, open(path, \"wb\"))").replace(
            "        opt = cls(None)\n        opt.k, opt.told = json.loads(open(path).read())\n        return opt",
            "        return pickle.load(open(path, \"rb\"))")
        self.assertIn("pickle.dump", source)
        script = self.script(source, "pickled.py")
        self.assertEqual(self.run_block(1, 2, script=script).error, None)
        self.assertEqual(self.run_block(2, 1, state="continue", script=script).error, None)
        self.assertEqual(self.store.history()[0][-1][0], 3)

    def test_host_death_is_reported(self):
        result = self.run_block(1, 2, script=self.script("import os\nos._exit(3)\n", "dead.py"))
        self.assertIn("exited with code 3", result.error)


    def test_indexed_pool_rejects_repeat(self):
        self.task = Pool()
        self.store = data.Store(self.task, self.dir / "pool.csv", self.dir / "private" / "pool.csv")
        data.write_space(self.task, self.dir / "space.json")
        result = self.run_block(1, 4, args={"repeat_after": 2}, script=self.script(POOL, "pool.py"))
        self.assertEqual(result.used, 2)
        self.assertIn("already evaluated", result.error)
        self.assertEqual(self.store.history()[0], [0, 1])


if __name__ == "__main__":
    unittest.main()
