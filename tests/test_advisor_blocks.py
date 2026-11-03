"""Block folders: open a block, propose it round by round, validate and accept."""
import json
from pathlib import Path
import tempfile
import textwrap
import unittest

from advisor_fixtures import PROBLEM, campaign, quiet
from agentoptimizer.advisor.blocks import (BlockError, accept_block, current_block, new_block, read_round,
                                           run_block, validate_proposals)
from agentoptimizer.advisor.cli import main
from agentoptimizer.advisor.data import load_campaign

ROWS = '''\
    1,0,done,import,import,100,0.01,5,0.42,
    2,1,done,advisor,sobol,150,0.1,12,0.61,
    3,1,pending,advisor,sobol,190,0.5,20,,
'''

# Three fixed points, taken in order across rounds.
PROPOSE = '''\
from agentoptimizer import batch

SIZE = {size}
POINTS = [[120, 0.05, 8], [130.5, 0.2, 9.0], [140, 0.3, 10]]

def propose(campaign, q):
    print('round', campaign.round)
    done = campaign.data.in_block(campaign.block)
    return batch.Proposal('gp_bo', campaign.problem.to_unit(POINTS[done:done + q]))
'''


def rounds_of(size):
    return PROBLEM.replace('block_size = 4', f'block_size = 4\nround_size = {size}')


class Blocks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = campaign(Path(self.tmp.name) / 'c', ROWS)

    def tearDown(self):
        self.tmp.cleanup()

    def proposals(self, text, block=2, size=20):
        path = self.folder / 'blocks' / f'{block:03d}'
        path.mkdir(parents=True, exist_ok=True)
        (path / 'proposals.csv').write_text(textwrap.dedent(text))
        q = len(text.strip().splitlines()) - 1
        (path / 'round.json').write_text(json.dumps({'size': size, 'round': 1, 'q': q}))

    def decide(self, number):
        with (self.folder / 'decisions.md').open('a') as f:
            f.write(f'## Block {number}\nGP-BO, because ...\n\n')

    def script(self, text, block=2):
        (self.folder / 'blocks' / f'{block:03d}' / 'propose.py').write_text(text)

    def test_new_block_saves_summary(self):
        number, folder = new_block(self.folder)
        self.assertEqual((number, folder), (2, self.folder / 'blocks/002'))
        summary = json.loads((folder / 'summary.json').read_text())
        self.assertEqual((summary['block'], summary['incumbent']['id']), (2, 2))
        self.assertIn('scipy', summary['environment'])
        self.assertIn('model', summary)  # None here: only one done row
        # Reopening an unaccepted block reuses its folder.
        self.assertEqual(new_block(self.folder)[0], 2)

    def test_first_block_of_an_empty_campaign_is_one(self):
        empty = campaign(Path(self.tmp.name) / 'empty')
        self.assertEqual(new_block(empty)[0], 1)

    def test_one_round_per_block(self):
        campaign(self.folder, problem=rounds_of(2))
        _, folder = new_block(self.folder)
        self.script(PROPOSE.format(size=2))
        rows, issues = run_block(self.folder)
        self.assertEqual((len(rows), issues), (2, []))
        self.assertIn('round 1', (folder / 'propose.log').read_text())
        self.assertEqual(read_round(self.folder, 2), {'size': 2, 'round': 1, 'q': 2})
        with self.assertRaisesRegex(BlockError, 'no "## Block 2" entry'):
            accept_block(self.folder)
        self.decide(2)
        self.assertEqual(accept_block(self.folder), [4, 5])
        problem, table, data = load_campaign(self.folder)
        self.assertEqual((table.rows[-1]['temperature'], table.rows[-1]['stir']), ('130.5', '9'))
        self.assertEqual((data.status[-2:].tolist(), data.blocks[-1], data.optimizer[-1]),
                         (['pending', 'pending'], 2, 'gp_bo'))
        self.assertEqual(current_block(self.folder), 3)
        with self.assertRaisesRegex(BlockError, 'not the block taking proposals'):
            accept_block(self.folder, 2)
        with self.assertRaisesRegex(BlockError, 'block 2 is complete'):
            run_block(self.folder, 2)
        self.assertEqual(new_block(self.folder)[0], 3)

    def test_rounds_fill_the_block(self):
        new_block(self.folder)
        self.script(PROPOSE.format(size=3))
        self.decide(2)
        for number in (1, 2, 3):
            rows, issues = run_block(self.folder)
            self.assertEqual((len(rows), issues), (1, []))
            self.assertEqual(read_round(self.folder, 2)['round'], number)
            self.assertEqual(accept_block(self.folder), [3 + number])
            if number < 3:
                self.assertEqual(current_block(self.folder), 2)
                with self.assertRaisesRegex(BlockError, 'still has rounds to run'):
                    new_block(self.folder)
        _, table, _ = load_campaign(self.folder)
        self.assertEqual([r['temperature'] for r in table.rows[3:]], ['120.0', '130.5', '140.0'])
        self.assertEqual(new_block(self.folder)[0], 3)

    def test_script_contract(self):
        new_block(self.folder)
        cases = [
            ('raise SystemExit("no idea")\n', 'exited with 1(.|\n)*no idea'),
            ('def propose(campaign, q):\n    pass\n', 'must set SIZE'),
            ('SIZE = 2\n', 'must define propose'),
            ('SIZE = 2\ndef propose(campaign, q):\n    return None\n', 'must return a batch.Proposal of 1'),
            (PROPOSE.format(size=18), 'SIZE = 18 exceeds the remaining budget of 17'),
        ]
        folder = self.folder / 'blocks/002'
        for text, expected in cases:
            with self.subTest(expected=expected):
                (folder / 'proposals.csv').write_text('stale\n')
                self.script(text)
                with self.assertRaisesRegex(BlockError, expected):
                    run_block(self.folder)
                self.assertFalse((folder / 'proposals.csv').exists())

    def test_unseeded_script_is_refused(self):
        new_block(self.folder)
        self.script('import numpy as np\nfrom agentoptimizer import batch\nSIZE = 1\n'
                    'def propose(campaign, q):\n'
                    '    return batch.Proposal("random", np.random.random((q, 3)))\n')
        with self.assertRaisesRegex(BlockError, 'different proposals on a second run'):
            run_block(self.folder)
        self.assertEqual(run_block(self.folder, verify=False)[1], [])

    def test_validation_errors(self):
        header = 'temperature,conc,stir,optimizer\n'
        cases = [
            (header + '250,0.1,3,m\n', 'line 2, temperature: 250 is outside'),
            (header + '100,0.1,3.5,m\n', 'stir: 3.5 is not a whole number'),
            (header + '100,0.1,3,m\n100,0.1,3,m\n', 'line 3: repeats line 2'),
            (header + '190,0.5,20,m\n', 'repeats id 3, which is still pending'),
            ('temperature,conc,stir,optimizer,colour\n100,0.1,3,m,red\n', 'unknown column(s) colour'),
            ('temperature,conc,optimizer\n100,0.1,m\n', 'missing column(s) stir'),
            ('temperature,conc,stir\n100,0.1,3\n', 'missing column(s) optimizer'),
            (header + '100,0.1,3,\n', 'optimizer: is empty'),
            (header, 'has no rows'),
            (header + ''.join(f'{80 + i},0.1,3,m\n' for i in range(18)),
             '18 proposals exceed the remaining budget of 17'),
        ]
        for text, expected in cases:
            with self.subTest(expected=expected):
                self.proposals(text)
                _, issues = validate_proposals(self.folder)
                self.assertTrue(any(expected in str(i) and i.level == 'error' for i in issues), issues)
        self.proposals(header + '100,0.1,3,m\n', size=0)
        self.assertIn('past its SIZE of 0', str(validate_proposals(self.folder)[1][0]))
        (self.folder / 'blocks/002/round.json').write_text(json.dumps({'size': 5, 'round': 1, 'q': 2}))
        self.assertIn('has 1 rows; this round has 2', str(validate_proposals(self.folder)[1][0]))
        (self.folder / 'blocks/002/round.json').unlink()
        self.assertIn('round.json: not found', str(validate_proposals(self.folder)[1][0]))
        self.proposals(header + '150,0.1,12,m\n')
        _, issues = validate_proposals(self.folder)
        self.assertEqual([i.level for i in issues], ['warning'])
        self.assertIn('repeats id 2 (a replicate)', str(issues[0]))

    def test_cli_flow(self):
        stack, out = quiet()
        with stack:
            self.assertEqual(main(['new-block', str(self.folder)]), 0)
            self.script(PROPOSE.format(size=2))
            self.assertEqual(main(['run', str(self.folder)]), 0)
            self.assertEqual(main(['accept', str(self.folder)]), 1)
            self.decide(2)
            self.assertEqual(main(['accept', str(self.folder)]), 0)
            self.assertEqual(main(['run', str(self.folder)]), 0)
            self.assertEqual(main(['accept', str(self.folder)]), 0)
        text = out.getvalue()
        self.assertIn('Block 2 is open', text)
        self.assertIn('Block 2, round 1: 1 of SIZE 2.', text)
        self.assertIn('Added ids 4-4', text)
        self.assertIn('Block 2 has 1 of 2; after recording, run and accept its next round.', text)
        self.assertIn('Block 2 is complete.', text)


if __name__ == '__main__':
    unittest.main()
