"""Block folders: open, run propose.py, validate and accept proposals."""
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

from advisor_fixtures import campaign, quiet
from agentoptimizer.advisor.blocks import (BlockError, accept_block, new_block, run_block,
                                           validate_proposals)
from agentoptimizer.advisor.cli import main
from agentoptimizer.advisor.data import load_campaign

ROWS = '''\
    1,0,done,import,import,100,0.01,5,0.42,
    2,1,done,advisor,sobol,150,0.1,12,0.61,
    3,1,pending,advisor,sobol,190,0.5,20,,
'''

PROPOSE = '''\
import csv
rows = [{'temperature': 120, 'conc': 0.05, 'stir': 8, 'optimizer': 'gp_bo'},
        {'temperature': 130.5, 'conc': 0.2, 'stir': 9.0, 'optimizer': 'gp_bo'}]
with open('proposals.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('wrote', len(rows))
'''


class Blocks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = campaign(Path(self.tmp.name) / 'c', ROWS)

    def tearDown(self):
        self.tmp.cleanup()

    def proposals(self, text, block=2):
        path = self.folder / 'blocks' / f'{block:03d}'
        path.mkdir(parents=True, exist_ok=True)
        (path / 'proposals.csv').write_text(textwrap.dedent(text))

    def decide(self, number):
        with (self.folder / 'decisions.md').open('a') as f:
            f.write(f'## Block {number}\nGP-BO, 2 points, because ...\n\n')

    def test_new_block_saves_summary(self):
        number, folder = new_block(self.folder)
        self.assertEqual((number, folder), (2, self.folder / 'blocks/002'))
        summary = json.loads((folder / 'summary.json').read_text())
        self.assertEqual((summary['block'], summary['incumbent']['id']), (2, 2))
        # Reopening an unaccepted block reuses its folder.
        self.assertEqual(new_block(self.folder)[0], 2)

    def test_first_block_of_an_empty_campaign_is_one(self):
        empty = campaign(Path(self.tmp.name) / 'empty')
        self.assertEqual(new_block(empty)[0], 1)

    def test_run_validate_accept(self):
        _, folder = new_block(self.folder)
        (folder / 'propose.py').write_text(PROPOSE)
        rows, issues = run_block(self.folder)
        self.assertEqual((len(rows), issues), (2, []))
        self.assertIn('wrote 2', (folder / 'propose.log').read_text())
        with self.assertRaisesRegex(BlockError, 'no "## Block 2" entry'):
            accept_block(self.folder)
        self.decide(2)
        self.assertEqual(accept_block(self.folder), [4, 5])
        problem, table, data = load_campaign(self.folder)
        self.assertEqual(table.rows[-1]['temperature'], '130.5')
        self.assertEqual(table.rows[-1]['stir'], '9')
        self.assertEqual((data.status[-2:].tolist(), data.blocks[-1], data.optimizer[-1]),
                         (['pending', 'pending'], 2, 'gp_bo'))
        with self.assertRaisesRegex(BlockError, 'already in data.csv'):
            accept_block(self.folder, 2)
        # The next block is now 3.
        self.assertEqual(new_block(self.folder)[0], 3)

    def test_failing_script_and_stale_output(self):
        _, folder = new_block(self.folder)
        (folder / 'proposals.csv').write_text('temperature,conc,stir\n100,0.1,3\n')
        (folder / 'propose.py').write_text('raise SystemExit("no idea")\n')
        with self.assertRaisesRegex(BlockError, 'exited with 1(.|\n)*no idea'):
            run_block(self.folder)
        self.assertFalse((folder / 'proposals.csv').exists())
        (folder / 'propose.py').write_text('print("forgot to write")\n')
        self.assertIn('not found', str(run_block(self.folder)[1][0]))

    def test_validation_errors(self):
        cases = [
            ('temperature,conc,stir\n250,0.1,3\n', 'line 2, temperature: 250 is outside'),
            ('temperature,conc,stir\n100,0.1,3.5\n', 'stir: 3.5 is not a whole number'),
            ('temperature,conc,stir\n100,0.1,3\n100,0.1,3\n', 'line 3: repeats line 2'),
            ('temperature,conc,stir\n190,0.5,20\n', 'repeats id 3, which is still pending'),
            ('temperature,conc,stir,colour\n100,0.1,3,red\n', 'unknown column(s) colour'),
            ('temperature,conc\n100,0.1\n', 'missing column(s) stir'),
            ('temperature,conc,stir,optimizer\n100,0.1,3,\n', 'optimizer: is empty'),
            ('temperature,conc,stir\n', 'has no rows'),
            ('temperature,conc,stir\n' + ''.join(f'{80 + i},0.1,3\n' for i in range(18)),
             '18 proposals exceed the remaining budget of 17'),
        ]
        for text, expected in cases:
            with self.subTest(expected=expected):
                self.proposals(text)
                _, issues = validate_proposals(self.folder)
                self.assertTrue(any(expected in str(i) and i.level == 'error' for i in issues), issues)
        self.proposals('temperature,conc,stir\n150,0.1,12\n')
        _, issues = validate_proposals(self.folder)
        self.assertEqual([i.level for i in issues], ['warning'])
        self.assertIn('repeats id 2 (a replicate)', str(issues[0]))

    def test_accept_needs_a_label(self):
        self.proposals('temperature,conc,stir\n100,0.1,3\n')
        self.decide(2)
        with self.assertRaisesRegex(BlockError, 'no optimizer column'):
            accept_block(self.folder)
        self.assertEqual(accept_block(self.folder, optimizer='manual_grid'), [4])

    def test_cli_flow(self):
        stack, out = quiet()
        with stack:
            self.assertEqual(main(['new-block', str(self.folder)]), 0)
            (self.folder / 'blocks/002/propose.py').write_text(PROPOSE)
            self.assertEqual(main(['run', str(self.folder)]), 0)
            self.assertEqual(main(['accept', str(self.folder)]), 1)
            self.decide(2)
            self.assertEqual(main(['accept', str(self.folder)]), 0)
        text = out.getvalue()
        self.assertIn('Block 2 is open', text)
        self.assertIn('2 proposals:', text)
        self.assertIn('Added ids 4-5', text)


if __name__ == '__main__':
    unittest.main()
