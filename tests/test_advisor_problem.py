"""problem.toml: parsing, strict validation and unit-cube scaling."""
import contextlib
import io
from pathlib import Path
import tempfile
import tomllib
import unittest

import numpy as np

from agentoptimizer.advisor.cli import main
from agentoptimizer.advisor.problem import TEMPLATE, ProblemError, load_problem, parse_problem


def spec(**changes):
    data = {'objective': {'name': 'yield', 'direction': 'maximize'},
            'parameters': [{'name': 'temperature', 'low': 80, 'high': 200},
                           {'name': 'concentration', 'low': .001, 'high': 1., 'log': True},
                           {'name': 'stir', 'type': 'integer', 'low': 1, 'high': 30}]}
    data.update(changes)
    return data


class Parsing(unittest.TestCase):
    def test_template_is_valid(self):
        problem = parse_problem(tomllib.loads(TEMPLATE))
        self.assertEqual(problem.names, ['temperature', 'concentration', 'stir_minutes'])
        self.assertEqual((problem.objective, problem.sign, problem.total_budget), ('yield', -1., 40))

    def test_defaults(self):
        problem = parse_problem(spec(), default_name='folder')
        self.assertEqual((problem.name, problem.noise, problem.total_budget), ('folder', 'unknown', None))
        self.assertEqual(problem.parameters[0].type, 'continuous')

    def test_errors_name_the_problem(self):
        cases = [
            (spec(colour='blue'), 'unknown key'),
            (spec(objective={'name': 'yield'}), 'direction'),
            (spec(objective={'name': 'id', 'direction': 'minimize'}), 'reserved'),
            (spec(parameters=[]), 'at least one'),
            (spec(parameters=[{'name': 'a', 'low': 1, 'high': 1}]), 'below high'),
            (spec(parameters=[{'name': 'a', 'low': 0, 'high': 1, 'log': True}]), 'low > 0'),
            (spec(parameters=[{'name': 'a', 'type': 'integer', 'low': .5, 'high': 3}]), 'whole-number'),
            (spec(parameters=[{'name': 'a', 'type': 'integer', 'low': 1, 'high': 3, 'log': True}]), 'continuous'),
            (spec(parameters=[{'name': 'a', 'low': 0, 'high': 1, 'step': 1}]), 'unknown key'),
            (spec(parameters=[{'name': 'a', 'low': 0, 'high': 1}, {'name': 'a', 'low': 0, 'high': 1}]), 'unique'),
            (spec(budget={'total': 4, 'block_size': 8}), 'exceeds'),
            (spec(budget={'total': 0}), 'positive integer'),
            (spec(noise='medium'), 'noise'),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ProblemError, message):
                parse_problem(data)


class Scaling(unittest.TestCase):
    def test_round_trip_and_ranges(self):
        problem = parse_problem(spec())
        X = np.array([[80., .001, 1.], [200., 1., 30.], [140., .0316, 15.]])
        U = problem.to_unit(X)
        self.assertTrue(((U >= 0) & (U <= 1)).all())
        np.testing.assert_allclose(problem.from_unit(U), X)
        # Log scale puts the geometric midpoint at 0.5.
        self.assertAlmostEqual(problem.to_unit([[80., 10 ** -1.5, 1.]])[0, 1], .5)

    def test_integers_get_equal_shares(self):
        problem = parse_problem(spec(parameters=[{'name': 'n', 'type': 'integer', 'low': 1, 'high': 4}]))
        values = problem.from_unit(np.linspace(0, 1, 4001)[:, None])[:, 0]
        counts = np.bincount(values.astype(int))[1:]
        self.assertEqual(counts.tolist(), [1000, 1000, 1000, 1001])
        self.assertTrue(problem.from_unit([[1.5], [-1.]])[:, 0].tolist() == [4., 1.])


class Cli(unittest.TestCase):
    def test_init_then_check(self):
        quiet = contextlib.ExitStack()
        quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
        quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))
        with quiet, tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'campaign'
            self.assertEqual(main(['init', str(folder)]), 0)
            self.assertEqual(load_problem(folder).dim, 3)
            self.assertEqual(main(['check', str(folder)]), 0)
            with self.assertRaises(SystemExit):
                main(['init', str(folder)])
            (folder / 'problem.toml').write_text('name = "x"\n')
            self.assertEqual(main(['check', str(folder)]), 1)


if __name__ == '__main__':
    unittest.main()
