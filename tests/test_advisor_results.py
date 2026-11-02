"""Results in: recording pending rows, manual experiments and imports."""
from pathlib import Path
import re
import tempfile
import unittest

from advisor_fixtures import campaign, quiet
from agentoptimizer.advisor.blocks import new_block
from agentoptimizer.advisor.cli import main
from agentoptimizer.advisor.data import load_campaign
from agentoptimizer.advisor.results import ResultsError, import_rows, record_results

ROWS = '''\
    1,0,done,import,import,100,0.01,5,0.42,
    2,1,done,advisor,sobol,150,0.1,12,0.61,
    3,2,pending,advisor,gp_bo,190,0.5,20,,
    4,2,pending,advisor,gp_bo,120,0.05,8,,
'''


class Results(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = campaign(Path(self.tmp.name) / 'c', ROWS)

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, text):
        path = Path(self.tmp.name) / 'results.csv'
        path.write_text(text)
        return path

    def test_record_pending_rows_and_a_manual_one(self):
        path = self.file('id,yield,status,notes,temperature\n'
                         '3,0.70,,good film,190\n'
                         '4,,failed,cracked,\n'
                         ',0.55,,tried by hand,\n')
        with self.assertRaisesRegex(ResultsError, 'line 4: id 5|nothing was written'):
            record_results(self.folder, path)  # the manual row has no parameters
        path = self.file('id,yield,status,notes,temperature,conc,stir\n'
                         '3,0.70,,good film,190,,\n'
                         '4,,failed,cracked,,,\n'
                         ',0.55,,tried by hand,110,0.02,4\n')
        self.assertEqual(record_results(self.folder, path), ([3, 4], [5]))
        _, table, data = load_campaign(self.folder)
        self.assertEqual(data.status.tolist(), ['done', 'done', 'done', 'failed', 'done'])
        self.assertEqual((data.y[2], table.rows[2]['notes'], table.rows[3]['notes']), (.70, 'good film', 'cracked'))
        manual = table.rows[4]
        self.assertEqual((manual['block'], manual['source'], manual['optimizer']), ('2', 'manual', 'manual'))
        self.assertEqual(new_block(self.folder)[0], 3)

    def test_bad_results_change_nothing(self):
        before = (self.folder / 'data.csv').read_text()
        cases = [
            ('id,yield\n9,0.5\n', 'data.csv has no id 9'),
            ('id,yield\n2,0.5\n', 'id 2 is done, not pending'),
            ('id,yield\n3,\n', 'give a yield value, or status failed'),
            ('id,yield\n3,0.5\n3,0.6\n', 'id 3 appears twice'),
            ('id,yield,temperature\n3,0.5,180\n', 'has temperature 180 here but 190 in data.csv'),
            ('id,yield,status\n3,0.5,failed\n', 'a failed row has no result'),
            ('id,yield\n3,abc\n', 'a done row needs a finite value'),
            ('id,score\n3,0.5\n', 'unknown column(s) score'),
        ]
        for text, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ResultsError, re.escape(expected)):
                record_results(self.folder, self.file(text))
            self.assertEqual((self.folder / 'data.csv').read_text(), before)

    def test_import(self):
        empty = campaign(Path(self.tmp.name) / 'empty')
        path = self.file('temperature,conc,stir,yield,notes\n100,0.01,5,0.42,old run\n120,0.02,6,,\n')
        with self.assertRaisesRegex(ResultsError, 'line 3: give a yield value'):
            import_rows(empty, path)
        path = self.file('temperature,conc,stir,yield,status,notes\n100,0.01,5,0.42,,old run\n120,0.02,6,,failed,\n')
        self.assertEqual(import_rows(empty, path), [1, 2])
        _, table, data = load_campaign(empty)
        self.assertEqual((data.blocks.tolist(), data.status.tolist()), ([0, 0], ['done', 'failed']))
        self.assertEqual((table.rows[0]['source'], table.rows[0]['optimizer']), ('import', 'import'))
        self.assertEqual(new_block(empty)[0], 1)
        for text, expected in [('id,temperature,conc,stir,yield\n1,100,0.01,5,0.4\n', 'import assigns ids'),
                               ('temperature,conc,yield\n100,0.01,0.4\n', 'missing column(s) stir'),
                               ('temperature,conc,stir,yield,colour\n100,0.01,5,0.4,red\n', 'unknown column(s) colour'),
                               ('temperature,conc,stir,yield\n300,0.01,5,0.4\n', '300 is outside')]:
            with self.subTest(expected=expected), self.assertRaisesRegex(ResultsError, re.escape(expected)):
                import_rows(empty, self.file(text))

    def test_cli(self):
        stack, out = quiet()
        with stack:
            self.assertEqual(main(['record', str(self.folder), str(self.file('id,yield\n3,0.7\n4,0.2\n'))]), 0)
            self.assertEqual(main(['record', str(self.folder), str(self.file('id,yield\n3,0.7\n'))]), 1)
        self.assertIn('Recorded 2 pending row(s) (ids 3-4).', out.getvalue())


if __name__ == '__main__':
    unittest.main()
