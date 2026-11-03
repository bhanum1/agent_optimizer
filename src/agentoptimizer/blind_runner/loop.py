"""The loop: advisor, executor and lab, one block at a time.

The runner plays the lab. Each block, a fresh advisor call in the sandbox
decides the next block in workspace/campaign/ exactly as /opt-run does and
accepts its first round; the loop checks it, reproduces it, evaluates it
privately and records the results. It then proposes the block's later rounds
itself, by running the block's propose.py in the sandbox on the data so far,
until the block is full. See docs/plan/06-unified-loop.md.

private/campaign/ holds the truth: problem.toml, candidates.csv, data.csv and
decisions.md as of the last evaluated round. The workspace copy is rewritten
from it before every call, and everything read back from the workspace is
checked against it.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid

from ..advisor.blocks import OUTPUTS, block_folder, current_block, has_decision
from ..advisor.data import DataError, format_number, load_campaign, read_table, write_table
from ..advisor.problem import ProblemError, load_problem

# What the runner records about each advisor call, summed over a block's attempts.
USAGE = ('turns', 'tool_calls', 'seconds', 'cost_usd',
         'input_tokens', 'cache_write_tokens', 'cache_read_tokens', 'output_tokens')

# Files of a public campaign the runner reads back, and their size limits.
LIMITS = {'data.csv': 20_000_000, 'decisions.md': 1_000_000, 'block file': 1_000_000}


class LoopError(RuntimeError):
    """The advisor's output broke a rule of the loop; the campaign stops here.
    `usage` is that call's usage, when the call itself broke the rule."""

    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage or {}


def label_for(block, method):
    """The evaluator's optimizer label for a row: block number and method."""
    return re.sub(r'[^A-Za-z0-9_.+ -]', '_', f'block {block} {method}')[:100]


class Loop:
    """One campaign. `advise(block, attempt, logs)` makes one advisor call
    and returns its usage; `sandbox(command)` runs a shell command in the
    workspace under the agent's isolation and returns a CompletedProcess."""

    def __init__(self, workspace, private, evaluator, advise, sandbox,
                 python='.venv/bin/python', retries=1, campaign='run'):
        self.workspace = Path(workspace).resolve()
        self.private = Path(private).resolve()
        self.public = self.workspace / 'campaign'
        self.truth = self.private / 'campaign'
        self.evaluator, self.advise, self.sandbox = evaluator, advise, sandbox
        self.python, self.retries, self.campaign = python, retries, campaign
        self.problem = load_problem(self.truth)
        self.records = []

    # Files in the workspace. The advisor could have replaced any of them with
    # a link, so the runner never follows one and caps what it reads.
    def _check_path(self, path):
        for part in [path, *path.parents]:
            if part == self.workspace:
                return
            if part.is_symlink():
                raise LoopError(f'{part.relative_to(self.workspace)} is a link')
        raise LoopError(f'{path} is outside the workspace')

    def read_public(self, path, limit):
        self._check_path(path)
        if not path.exists():
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise LoopError(f'{path.relative_to(self.workspace)} is not a regular file')
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise LoopError(f'{path.relative_to(self.workspace)} is larger than {limit} bytes')
        return data.decode()

    def write_public(self, path, text):
        self._check_path(path.parent)
        temporary = path.with_name(f'.{uuid.uuid4().hex}.tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(temporary, path)

    def remaining(self):
        return self.problem.total_budget - len(read_table(self.truth, self.problem).rows)

    def run(self):
        """Advise, check and evaluate blocks until the budget is used or a block fails."""
        records = self.records = []
        while self.remaining() > 0:
            number = current_block(self.truth)
            records.append({'block': number, 'attempts': 0, 'status': 'no proposals', 'size': 0, 'rounds': 0,
                            'rows': 0, 'methods': '', 'propose_sha256': '', **dict.fromkeys(USAGE, 0)})
            try:
                self.block(records[-1])
            except LoopError as exc:
                # The block's calls were made and paid for; keep their record.
                records[-1]['status'] = 'refused'
                for key in USAGE:
                    records[-1][key] += exc.usage.get(key, 0)
                raise
            finally:
                self.write_records(records)
            if records[-1]['status'] != 'evaluated':
                break
        return records

    def publish(self):
        for name in ['problem.toml', 'candidates.csv', 'data.csv']:
            if (self.truth / name).exists():
                self.write_public(self.public / name, (self.truth / name).read_text())

    def block(self, record):
        """Advise, check, reproduce and evaluate one block, round by round, filling in its record."""
        number = record['block']
        logs = self.private / 'blocks' / f'{number:03d}'
        rows = []
        for attempt in range(1 + self.retries):
            self.publish()
            usage = self.advise(number, attempt, logs / f'attempt{attempt + 1}') or {}
            record['attempts'] += 1
            for key in USAGE:
                record[key] += usage.get(key, 0)
            rows, decisions = self.collect(number)
            if rows:
                break
        if not rows:
            return
        files = self.block_files(number, logs)
        size = self.reproduce(number, files, rows, decisions, logs)
        methods = []
        while rows:
            self.commit(rows, self.evaluate(number, rows, decisions if not record['rounds'] else None), decisions)
            record['rounds'] += 1
            methods += [row['optimizer'] for row in rows]
            done = load_campaign(self.truth)[2].in_block(number)
            rows = self.next_round(number, files, decisions, size, logs) if done < size and self.remaining() else []
        record.update(status='evaluated', size=size, rows=len(methods), methods=' + '.join(dict.fromkeys(methods)),
                      propose_sha256=hashlib.sha256(files['propose.py'].encode()).hexdigest())

    def collect(self, number):
        """The block's first round as the advisor accepted it, and decisions.md, checked against the truth."""
        decisions = self.read_public(self.public / 'decisions.md', LIMITS['decisions.md']) or ''
        if not decisions.startswith((self.truth / 'decisions.md').read_text()):
            raise LoopError('decisions.md changed earlier entries; it is append-only')
        rows = self.new_rows(self.public / 'data.csv', number, decisions)
        if rows and not has_decision(self.private / 'inbox', number):
            raise LoopError(f'decisions.md has no entry for block {number}')
        return rows, decisions

    def new_rows(self, path, number, decisions):
        """Rows a data.csv in the workspace adds to the truth: only pending advisor rows of
        the block, within the budget; every recorded row and column unchanged."""
        inbox = self.private / 'inbox'
        shutil.rmtree(inbox, ignore_errors=True)
        shutil.copytree(self.truth, inbox)
        text = self.read_public(path, LIMITS['data.csv'])
        if text is None:
            raise LoopError(f'{path.relative_to(self.workspace)} was removed')
        (inbox / 'data.csv').write_text(text)
        (inbox / 'decisions.md').write_text(decisions)
        try:
            _, table, _ = load_campaign(inbox)
        except (DataError, ProblemError) as exc:
            raise LoopError(f'{path.relative_to(self.workspace)} is not valid: {exc}') from None
        truth = read_table(self.truth, self.problem)
        if table.columns != truth.columns or table.rows[:len(truth.rows)] != truth.rows:
            raise LoopError(f'{path.relative_to(self.workspace)} changed rows or columns that were already recorded')
        rows = table.rows[len(truth.rows):]
        for row in rows:
            if (row['status'], row['block'], row['source']) != ('pending', str(number), 'advisor'):
                raise LoopError(f'{path.relative_to(self.workspace)} row {row["id"]} is not a pending advisor row '
                                  f'of block {number}; only accepted proposals may be added')
        if len(rows) > self.remaining():
            raise LoopError(f'block {number} has {len(rows)} rows, over the remaining budget')
        return rows

    def block_files(self, number, logs):
        """The block folder's inputs as the advisor left them, kept for every round."""
        source = block_folder(self.public, number)
        self._check_path(source)
        if not source.is_dir():
            raise LoopError(f'campaign/blocks/{number:03d} is missing')
        files = {}
        for entry in sorted(source.iterdir()):
            if entry.is_file() and not entry.is_symlink():
                files[entry.name] = self.read_public(entry, LIMITS['block file'])
        if 'propose.py' not in files:
            raise LoopError(f'campaign/blocks/{number:03d}/propose.py is missing')
        logs.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (logs / name).write_text(text)
        return {name: text for name, text in files.items() if name not in OUTPUTS}

    def stage(self, number, files, decisions):
        """A copy of the truth campaign in work/ with the block's files, for the sandbox.
        The sandbox denies its TMPDIR (.tmp/) to commands; work/ is readable and writable."""
        verify = self.workspace / 'work/.verify'
        shutil.rmtree(verify, ignore_errors=True)
        folder = block_folder(verify / 'campaign', number)
        folder.mkdir(parents=True)
        for name in ['problem.toml', 'candidates.csv', 'data.csv']:
            if (self.truth / name).exists():
                shutil.copyfile(self.truth / name, verify / 'campaign' / name)
        (verify / 'campaign/decisions.md').write_text(decisions)
        for name, text in files.items():
            (folder / name).write_text(text)
        return verify, folder

    def reproduce(self, number, files, rows, decisions, logs):
        """Rerun the block's propose.py in the sandbox on the data the block saw;
        its first round must be the rows accepted. Returns the block's SIZE."""
        verify, folder = self.stage(number, files, decisions)
        result = self.sandbox(f'{self.python} -I opt_run.py run work/.verify/campaign '
                              f'--block {number} --no-verify')
        (logs / 'reproduce.log').write_text(result.stdout + result.stderr)
        output = self.read_public(folder / 'proposals.csv', LIMITS['data.csv'])
        planned = self.read_public(folder / 'round.json', LIMITS['block file'])
        if result.returncode or output is None or planned is None:
            raise LoopError(f'block {number}: propose.py did not run again; see {logs / "reproduce.log"}')
        proposals = list(csv.DictReader(io.StringIO(output)))
        if [self.identity(row) for row in proposals] != [self.identity(row) for row in rows]:
            raise LoopError(f'block {number}: the accepted rows are not what propose.py proposes '
                              'from the data the block saw')
        shutil.rmtree(verify)
        return int(json.loads(planned)['size'])

    def next_round(self, number, files, decisions, size, logs):
        """Propose and accept the block's next round in the sandbox, on the data so far."""
        data = load_campaign(self.truth)[2]
        expected = min(self.problem.round_size, size - data.in_block(number), self.remaining())
        verify, folder = self.stage(number, files, decisions)
        campaign = 'work/.verify/campaign'
        result = self.sandbox(f'{self.python} -I opt_run.py run {campaign} --block {number} --no-verify && '
                              f'{self.python} -I opt_run.py accept {campaign} --block {number}')
        log = logs / f'round{data.in_block(number) + 1:03d}.log'
        log.write_text(result.stdout + result.stderr)
        if result.returncode:
            raise LoopError(f'block {number}: a later round failed; see {log}')
        rows = self.new_rows(verify / 'campaign/data.csv', number, decisions)
        if len(rows) != expected:
            raise LoopError(f'block {number}: a round of {len(rows)} rows, not {expected}')
        shutil.rmtree(verify)
        return rows

    def identity(self, row):
        """What a row asks to evaluate: its candidate in a pool, else its values as accepted."""
        if self.problem.pool is not None:
            return row.get('candidate_id')
        return tuple(format_number(p, float(row[p.name])) for p in self.problem.parameters)

    def request(self, op, **kwargs):
        reply = self.evaluator.handle(self.campaign, {'request_id': uuid.uuid4().hex, 'op': op, **kwargs})
        if not reply['ok']:
            raise LoopError(f'evaluator refused {op}: {reply["error"]}')
        return reply['result']

    def evaluate(self, number, rows, decisions=None):
        """Evaluate the rows in order, journaling the decisions, if given, with the first."""
        values, label, documents = [], None, {'documents': {'decisions.md': decisions}} if decisions else {}
        for row in rows:
            if label_for(number, row['optimizer']) != label:
                label = label_for(number, row['optimizer'])
                self.request('label', name=label, **documents)
                documents = {}
            if self.problem.pool is not None:
                y = self.request('evaluate', candidate_index=int(row['candidate_id']))
            else:
                y = self.request('evaluate', x=[float(row[name]) for name in self.problem.names])
            values.append(float(y))
        return values

    def commit(self, rows, values, decisions):
        table = read_table(self.truth, self.problem)
        for row, value in zip(rows, values):
            table.rows.append({**row, 'status': 'done', self.problem.objective: repr(value)})
        write_table(self.truth, table)
        (self.truth / 'decisions.md').write_text(decisions)

    def write_records(self, records):
        with (self.private / 'blocks.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
