"""Candidate pools: experiments are chosen from candidates.csv by candidate_id."""
import json
from pathlib import Path
import re
import tempfile
import textwrap
import unittest

from advisor_fixtures import quiet
from agentoptimizer import batch
from agentoptimizer.advisor import Campaign, ProblemError
from agentoptimizer.advisor.blocks import accept_block, new_block, run_block, validate_proposals
from agentoptimizer.advisor.data import check_table, load_campaign, read_table
from agentoptimizer.advisor.problem import load_problem
from agentoptimizer.advisor.results import import_rows, record_results
from agentoptimizer.advisor.summary import format_summary, summarize

PROBLEM = '''\
[objective]
name = "score"
direction = "maximize"
[budget]
total = 14
[pool]
file = "candidates.csv"
[[parameters]]
name = "x1"
[[parameters]]
name = "x2"
type = "integer"
[[parameters]]
name = "x3"
'''


def features(i):
    """C028 and C029 share C000's features; x3 is constant."""
    j = 0 if i >= 28 else i
    return j * 10 / 27, j % 5, 7


def score(x1, x2, _):
    return -(x1 - 6) ** 2 + x2


def make(folder, problem=PROBLEM, candidates=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'problem.toml').write_text(problem)
    if candidates is None:
        candidates = 'candidate_id,x1,x2,x3,label\n' + ''.join(
            f'C{i:03d},{features(i)[0]!r},{features(i)[1]},{features(i)[2]},mat{i}\n' for i in range(30))
    (folder / 'candidates.csv').write_text(candidates)
    return folder


PROPOSE = '''
from agentoptimizer import batch

SIZE = 1

def propose(campaign, q):
    return batch.qlognei(**campaign.inputs(), q=q, seed=5)
'''


class Pools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = make(Path(self.tmp.name) / 'c')

    def tearDown(self):
        self.tmp.cleanup()

    def results(self, text):
        path = Path(self.tmp.name) / 'results.csv'
        path.write_text(text)
        return path

    def test_bounds_come_from_the_pool(self):
        problem = load_problem(self.folder)
        self.assertEqual(len(problem.pool), 30)
        self.assertEqual([(p.low, p.high) for p in problem.parameters], [(0., 10.), (0., 4.), (7., 8.)])

    def test_bad_pools(self):
        cases = [
            (PROBLEM.replace('total = 14', 'total = 31'), None, 'exceeds the 30 candidates'),
            (PROBLEM, 'candidate_id,x1,x2\nA,1,2\n', 'missing column(s) x3'),
            (PROBLEM, 'candidate_id,x1,x2,x3\nA,1,2,3\nA,2,2,3\n', 'repeated candidate_id(s) A'),
            (PROBLEM, 'candidate_id,x1,x2,x3\nA,1,2.5,3\nB,2,2,3\n', 'non-integer candidate values'),
            (PROBLEM, 'candidate_id,x1,x2,x3\nA,1,x,3\n', 'x2 must be a finite number'),
            (PROBLEM.replace('name = "x1"', 'name = "x1"\nlow = 2\nhigh = 5'), None, 'outside the bounds [2, 5]'),
            (PROBLEM.replace('file = "candidates.csv"', 'file = "missing.csv"'), None, 'not found'),
        ]
        for problem, candidates, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(ProblemError, re.escape(expected)):
                load_problem(make(Path(self.tmp.name) / 'bad', problem, candidates))

    def test_import_by_candidate_and_row_checks(self):
        rows = 'candidate_id,score\n' + ''.join(f'C{i:03d},{score(*features(i))}\n' for i in range(0, 20, 2))
        self.assertEqual(import_rows(self.folder, self.results(rows)), list(range(1, 11)))
        problem, table, data = load_campaign(self.folder)
        self.assertEqual(table.columns[:6], ['id', 'block', 'status', 'source', 'optimizer', 'candidate_id'])
        self.assertAlmostEqual(data.X[1, 0], 20 / 27)
        table.rows[0]['candidate_id'] = 'Z999'
        table.rows[1]['x1'] = '3.5'
        messages = [str(i) for i in check_table(problem, table)]
        self.assertTrue(any("candidate_id: 'Z999' is not in candidates.csv" in m for m in messages), messages)
        self.assertTrue(any('x1: 3.5 differs from candidates.csv' in m for m in messages), messages)

    def test_loop_with_pool_gp_bo(self):
        rows = 'candidate_id,score\n' + ''.join(f'C{i:03d},{score(*features(i))}\n' for i in range(0, 20, 2))
        import_rows(self.folder, self.results(rows))
        number, block = new_block(self.folder)
        (block / 'propose.py').write_text(textwrap.dedent(PROPOSE))
        (self.folder / 'decisions.md').write_text(f'## Block {number}\nPool GP-BO, one row.\n')
        proposals, issues = run_block(self.folder)
        self.assertEqual(issues, [])
        chosen = proposals[0]['candidate_id']
        self.assertNotIn(int(chosen[1:]), range(0, 20, 2))
        self.assertEqual(accept_block(self.folder), [11])
        _, table, _ = load_campaign(self.folder)
        self.assertEqual(table.rows[-1]['candidate_id'], chosen)
        record_results(self.folder, self.results(f'id,score\n11,{score(*features(int(chosen[1:])))}\n'))

        problem, _, data = load_campaign(self.folder)
        summary = summarize(problem, data)
        self.assertEqual(summary['pool'], {'file': 'candidates.csv', 'size': 30, 'tested': 11,
                                           'pending': 0, 'available': 19})
        self.assertIn('pool: 30 candidates in candidates.csv; 11 tested', format_summary(summary))
        self.assertIn('candidate C', format_summary(summary))
        self.assertNotIn('x3', [b['parameter'] for b in summary['incumbent']['at_bounds']])

    def test_proposal_checks(self):
        import_rows(self.folder, self.results('candidate_id,score\nC001,1\n'))
        (self.folder / 'blocks/001').mkdir(parents=True)

        def propose(text):
            (self.folder / 'blocks/001/proposals.csv').write_text(text)
            (self.folder / 'blocks/001/round.json').write_text(
                json.dumps({'size': 5, 'round': 1, 'q': len(text.splitlines()) - 1}))
        cases = [
            ('candidate_id,optimizer\nZ999,m\n', "'Z999' is not in candidates.csv"),
            ('candidate_id,optimizer\nC002,m\nC002,m\n', 'repeats line 2'),
            ('candidate_id,x1,optimizer\nC002,9,m\n', 'x1: 9 differs from candidates.csv'),
            ('x1,x2,x3,optimizer\n1,1,7,m\n', 'missing column(s) candidate_id'),
        ]
        for text, expected in cases:
            with self.subTest(expected=expected):
                propose(text)
                _, issues = validate_proposals(self.folder)
                self.assertTrue(any(expected in str(i) and i.level == 'error' for i in issues), issues)
        # Same features as C000 but a different candidate: not a repeat.
        propose('candidate_id,optimizer\nC000,m\nC028,m\n')
        self.assertEqual(validate_proposals(self.folder)[1], [])
        propose('candidate_id,optimizer\nC001,m\n')
        self.assertIn('repeats id 1 (a replicate)', str(validate_proposals(self.folder)[1][0]))

    def test_campaign_api_modes(self):
        import_rows(self.folder, self.results('candidate_id,score\nC001,1\n'))
        (self.folder / 'blocks/001').mkdir(parents=True)
        c = Campaign.load(self.folder / 'blocks/001')
        self.assertEqual((len(c.candidate_ids), len(c.available()), c.candidates().shape), (30, 29, (30, 3)))
        stack, _ = quiet()
        with stack:
            with self.assertRaisesRegex(ProblemError, 'pass candidates'):
                c.write(batch.Proposal('x', [[0, 0, 0]]))
            c.write(batch.Proposal('pick', c.candidates()[c.available()][[0, 5]], indices=[0, 5]))
        (self.folder / 'blocks/001/round.json').write_text(json.dumps({'size': 2, 'round': 1, 'q': 2}))
        self.assertEqual(validate_proposals(self.folder)[1], [])
        self.assertEqual([r.split(',')[0] for r in
                          (self.folder / 'blocks/001/proposals.csv').read_text().splitlines()[1:]], ['C000', 'C006'])
        box = Path(self.tmp.name) / 'box'
        box.mkdir()
        (box / 'problem.toml').write_text(PROBLEM.replace('[pool]\nfile = "candidates.csv"\n', '')
                                          .replace('name = "x1"', 'name = "x1"\nlow = 0\nhigh = 1')
                                          .replace('type = "integer"', 'type = "integer"\nlow = 0\nhigh = 4')
                                          .replace('name = "x3"', 'name = "x3"\nlow = 0\nhigh = 1'))
        with self.assertRaisesRegex(ProblemError, 'no \\[pool\\]'):
            Campaign.load(box).candidates()


if __name__ == '__main__':
    unittest.main()
