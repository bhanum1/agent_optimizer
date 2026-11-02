"""data.csv: reading, writing, row checks and parsing."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from advisor_fixtures import HEADER, campaign, quiet
from agentoptimizer.advisor.cli import main
from agentoptimizer.advisor.data import DataError, check_table, parse_table, read_table, write_table
from agentoptimizer.advisor.problem import load_problem

GOOD = '''\
    1,0,done,import,import,100,0.01,5,0.42,from the old notebook
    2,1,done,advisor,sobol,150,0.1,12,0.61,
    3,1,failed,advisor,sobol,190,0.9,30,,clogged
    4,2,pending,advisor,gp_bo,160,0.2,10,,
'''


class Tables(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, rows, header=HEADER):
        campaign(self.folder, rows, header=header)
        problem = load_problem(self.folder)
        return problem, read_table(self.folder, problem)

    def messages(self, rows, header=HEADER):
        problem, table = self.load(rows, header)
        return [str(issue) for issue in check_table(problem, table)]

    def test_valid_table_parses(self):
        problem, table = self.load(GOOD)
        self.assertEqual(check_table(problem, table), [])
        data = parse_table(problem, table)
        self.assertEqual(data.ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(data.where('done').tolist(), [True, True, False, False])
        np.testing.assert_allclose(data.X[1], [150, .1, 12])
        self.assertTrue(np.isnan(data.y[2:]).all())
        self.assertEqual(data.y[1], .61)

    def test_missing_file_is_an_empty_table(self):
        campaign(self.folder)
        problem = load_problem(self.folder)
        table = read_table(self.folder, problem)
        self.assertEqual((table.rows, table.columns[-2:]), ([], ['yield', 'notes']))
        self.assertEqual(len(parse_table(problem, table).ids), 0)

    def test_round_trip_keeps_extra_columns(self):
        header = HEADER.rstrip('\n') + ',operator\n'
        problem, table = self.load('1,0,done,import,import,100,0.01,5,0.42,,Ana\n', header)
        write_table(self.folder, table)
        self.assertEqual((self.folder / 'data.csv').read_text(), header + '1,0,done,import,import,100,0.01,5,0.42,,Ana\n')

    def test_each_error_names_line_and_column(self):
        cases = [
            ('x,0,done,import,import,100,0.01,5,0.4,', 'line 2, id: must be a positive whole number'),
            ('1,0,done,import,import,100,0.01,5,0.4,\n1,0,done,import,import,110,0.01,5,0.4,',
             'line 3 (id 1), id: repeats the id on line 2'),
            ('1,-1,done,import,import,100,0.01,5,0.4,', 'block: must be a whole number >= 0'),
            ('1,0,finished,import,import,100,0.01,5,0.4,', 'status: must be one of done, pending, failed'),
            ('1,0,done,lab,import,100,0.01,5,0.4,', 'source: must be one of'),
            ('1,0,done,import,,100,0.01,5,0.4,', 'optimizer: is empty'),
            ('1,0,done,import,import,250,0.01,5,0.4,', 'temperature: 250 is outside [80, 200]'),
            ('1,0,done,import,import,hot,0.01,5,0.4,', "temperature: must be a finite number, not 'hot'"),
            ('1,0,done,import,import,100,0.01,5.5,0.4,', 'stir: 5.5 is not a whole number'),
            ('1,0,done,import,import,100,0.01,5,,', 'yield: a done row needs a finite value'),
            ('1,0,done,import,import,100,0.01,5,nan,', 'yield: a done row needs a finite value'),
            ('1,0,pending,advisor,sobol,100,0.01,5,0.4,', 'yield: has a result but status pending'),
            ('1,0,failed,advisor,sobol,100,0.01,5,0.4,', 'yield: a failed row has no result'),
        ]
        for rows, expected in cases:
            with self.subTest(expected=expected):
                messages = self.messages(rows + '\n')
                self.assertTrue(any(expected in m for m in messages), messages)
                problem, table = self.load(rows + '\n')
                with self.assertRaises(DataError):
                    parse_table(problem, table)

    def test_missing_columns_and_bad_structure(self):
        header = 'id,block,status,source,optimizer,temperature,conc,yield\n'
        self.assertIn('missing column(s) stir', self.messages('1,0,done,import,import,100,0.01,0.4\n', header)[0])
        with self.assertRaisesRegex(DataError, 'duplicate column'):
            self.load('1,1,0\n', header='id,id,block\n')
        with self.assertRaisesRegex(DataError, 'line 2: 12 cells'):
            self.load('1,0,done,import,import,100,0.01,5,0.4,,x,y\n')

    def test_repeated_pending_point_is_a_warning(self):
        messages = self.messages('1,0,done,import,import,100,0.01,5,0.4,\n2,1,pending,advisor,sobol,100,0.01,5,,\n')
        self.assertEqual(messages, ['warning: line 3 (id 2): repeats the point of id 1, which is also pending or done'])

    def test_cli_check(self):
        campaign(self.folder, GOOD)
        stack, out = quiet()
        with stack:
            self.assertEqual(main(['check', str(self.folder)]), 0)
        self.assertIn('4 rows (2 done, 1 pending, 1 failed)', out.getvalue())
        campaign(self.folder, GOOD + '5,2,pending,advisor,gp_bo,300,0.2,10,,\n')
        stack, out = quiet()
        with stack:
            self.assertEqual(main(['check', str(self.folder)]), 1)
        self.assertIn('line 6 (id 5), temperature: 300 is outside', out.getvalue())


if __name__ == '__main__':
    unittest.main()
