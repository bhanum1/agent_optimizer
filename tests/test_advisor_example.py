"""The reaction example: its problem loads and its simulated lab records results."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

from agentoptimizer.advisor.data import load_campaign
from agentoptimizer.advisor.problem import load_problem

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/reaction'
spec = importlib.util.spec_from_file_location('reaction_lab', EXAMPLE / 'lab.py')
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class ReactionExample(unittest.TestCase):
    def test_problem_and_optimum(self):
        problem = load_problem(EXAMPLE)
        self.assertEqual((problem.names, problem.total_budget), (['temperature', 'catalyst', 'base_equiv', 'hours'], 40))
        self.assertAlmostEqual(lab.true_yield(85, 1.2, 2.4, 10), 95.)
        self.assertLess(lab.true_yield(40, .1, 1, 1), 10)

    def test_lab_measures_pending_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'reaction'
            folder.mkdir()
            shutil.copy(EXAMPLE / 'problem.toml', folder)
            self.assertIsNone(lab.measure(folder))
            (folder / 'data.csv').write_text(
                'id,block,status,source,optimizer,temperature,catalyst,base_equiv,hours,yield,notes\n'
                '1,1,pending,advisor,sobol,85,1.2,2.4,10,,\n'
                '2,1,pending,advisor,sobol,115,1.0,2.0,20,,\n')
            path = lab.measure(folder)
            self.assertEqual(path, folder / 'blocks/001/lab_results.csv')
            _, table, data = load_campaign(folder)
            self.assertEqual(data.status.tolist(), ['done', 'failed'])
            self.assertLess(abs(data.y[0] - 95), 5)
            self.assertEqual(table.rows[1]['notes'], 'black tar, no product peak')
            self.assertIsNone(lab.measure(folder))


if __name__ == '__main__':
    unittest.main()
