"""Private evaluator: durable budgets, idempotent requests and prospective decisions.

Only the trusted runner imports this module; the advisor never reaches it.
"""
import csv
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import traceback

from ..tasks.base import FEASIBILITY_TOL


# Decision documents a campaign journals; each may only grow.
DOCUMENTS = ('decisions.md',)


class BudgetExhausted(Exception):
    pass


def incumbents(db, campaign):
    """Each completed evaluation with its feasibility and the incumbent so far:
    the feasible point with the lowest observed y, and its noise-free value.
    Reads only, so it also serves a read-only connection to a live journal."""
    rows = db.execute('SELECT idx,x,y,g,y_true,label,elapsed,wall_ns FROM evaluations '
                      'WHERE campaign=? AND y IS NOT NULL ORDER BY idx', (campaign,)).fetchall()
    best, best_x, best_true = math.inf, None, None
    for row in rows:
        x, y, g, y_true = row[1:5]
        feasible = sum(max(v, 0.) for v in json.loads(g or '[]')) <= FEASIBILITY_TOL
        if feasible and y < best:
            best, best_x, best_true = y, json.loads(x), y_true
        yield row, feasible, (None if best_x is None else best), best_x, best_true


class Evaluator:
    def __init__(self, suite, database, task_factory):
        self.suite = suite
        self.lock = threading.Lock()
        self.tasks = {c['id']: task_factory(c) for c in suite['campaigns']}
        self.error_log = Path(database).parent / 'evaluator_errors.log'
        if isinstance(suite['budget'], bool) or not isinstance(suite['budget'], int) or suite['budget'] < 1:
            raise ValueError('Budget must be a positive integer.')
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS requests(campaign TEXT, id TEXT, digest TEXT, reply TEXT,
              PRIMARY KEY(campaign,id));
            CREATE TABLE IF NOT EXISTS evaluations(campaign TEXT, idx INTEGER, request_id TEXT,
              x TEXT, y REAL, label TEXT, elapsed REAL, wall_ns INTEGER,
              g TEXT, y_true REAL, candidate_index INTEGER, PRIMARY KEY(campaign,idx));
            CREATE TABLE IF NOT EXISTS labels(campaign TEXT PRIMARY KEY, label TEXT);
            CREATE TABLE IF NOT EXISTS documents(campaign TEXT, name TEXT, content TEXT,
              PRIMARY KEY(campaign,name));
            CREATE TABLE IF NOT EXISTS snapshots(campaign TEXT, eval_idx INTEGER, name TEXT,
              content TEXT, wall_ns INTEGER);
        ''')
        if 'candidate_index' not in {row[1] for row in self.db.execute('PRAGMA table_info(evaluations)')}:
            self.db.execute('ALTER TABLE evaluations ADD COLUMN candidate_index INTEGER')
        self.db.execute('CREATE TABLE IF NOT EXISTS configuration(value TEXT NOT NULL)')
        config = json.dumps(suite, sort_keys=True)
        previous = self.db.execute('SELECT value FROM configuration').fetchone()
        if previous and previous[0] != config:
            self.db.close()
            raise ValueError('Cannot change a saved campaign configuration or budget.')
        if not previous:
            self.db.execute('INSERT INTO configuration VALUES(?)', (config,))
        self.db.commit()

    def count(self, campaign):
        return self.db.execute('SELECT COUNT(*) FROM evaluations WHERE campaign=?', (campaign,)).fetchone()[0]

    def status(self, campaign):
        used = self.count(campaign)
        task = self.tasks[campaign]
        state = {'dim': int(task.dim), 'bounds': [[float(v) for v in pair] for pair in task.bounds],
                'direction': 'min', 'noisy': bool(getattr(task, 'noisy', False)),
                'n_constraints': int(getattr(task, 'n_constraints', 0)),
                'seed': self.suite.get('seed', 0),
                'budget': self.suite['budget'], 'evaluations_used': used,
                'evaluations_remaining': self.suite['budget'] - used}
        if getattr(task, 'indexed_pool', False):
            state.update(indexed_pool=True, n_candidates=int(task.n_candidates))
        return state

    def documents(self, campaign, docs):
        if not isinstance(docs, dict) or set(docs) != set(DOCUMENTS):
            raise ValueError('Invalid decision documents.')
        for name, content in docs.items():
            if not isinstance(content, str) or len(content.encode()) > 200000:
                raise ValueError('Invalid decision document size.')
            old = self.db.execute('SELECT content FROM documents WHERE campaign=? AND name=?',
                                  (campaign, name)).fetchone()
            old = old[0] if old else ''
            if not content.startswith(old):
                raise ValueError(f'{name} must be append-only.')
            if content != old:
                self.db.execute('INSERT OR REPLACE INTO documents VALUES(?,?,?)', (campaign, name, content))
                self.db.execute('INSERT INTO snapshots VALUES(?,?,?,?,?)',
                                (campaign, self.count(campaign), name, content, time.time_ns()))

    def incumbents(self, campaign):
        return incumbents(self.db, campaign)

    def scored_trace(self, campaign):
        """Private scoring trace in the baseline convention: y_best_so_far is the
        incumbent's noise-free value for noisy tasks, blank before the first feasible point."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['eval_idx', 'y', 'y_true', 'feasible', 'y_best_so_far'])
        for (idx, _, y, _, y_true, *_), feasible, best, _, best_true in self.incumbents(campaign):
            best = best_true if best_true is not None else best
            writer.writerow([idx, y, '' if y_true is None else y_true, feasible, '' if best is None else best])
        return output.getvalue()

    def handle(self, campaign, request):
        with self.lock:
            return self._handle(campaign, request)

    def _handle(self, campaign, request):
        if campaign not in self.tasks or not isinstance(request, dict):
            raise ValueError('Invalid campaign or request.')
        key = request.get('request_id')
        if not isinstance(key, str) or not re.fullmatch('[0-9a-f]{32}', key):
            raise ValueError('Invalid request identifier.')
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        old = self.db.execute('SELECT digest,reply FROM requests WHERE campaign=? AND id=?',
                              (campaign, key)).fetchone()
        if old:
            if not hmac.compare_digest(old[0], digest):
                raise ValueError('Request identifier was reused with different content.')
            if old[1] is None:
                raise ValueError('A previous request has an unknown outcome; stop and report.')
            return json.loads(old[1])
        self.db.execute('INSERT INTO requests VALUES(?,?,?,NULL)', (campaign, key, digest))
        self.db.commit()
        try:
            if 'documents' in request:
                self.documents(campaign, request['documents'])
            op = request.get('op')
            if op == 'status':
                result = self.status(campaign)
            elif op == 'label':
                label = request.get('name')
                if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9_.+ -]{1,100}', label):
                    raise ValueError('Invalid optimizer label.')
                self.db.execute('INSERT OR REPLACE INTO labels VALUES(?,?)', (campaign, label))
                result = None
            elif op == 'evaluate':
                n = self.count(campaign)
                if n >= self.suite['budget']:
                    raise BudgetExhausted('Evaluation budget exhausted.')
                if self.db.execute('SELECT 1 FROM evaluations WHERE campaign=? AND y IS NULL', (campaign,)).fetchone():
                    raise ValueError('An evaluation has an unknown outcome; stop and report.')
                task = self.tasks[campaign]
                indexed = getattr(task, 'indexed_pool', False)
                candidate_index = None
                if indexed:
                    if 'x' in request:
                        raise ValueError('Select an unused candidate by integer index.')
                    candidate_index = request.get('candidate_index')
                    task.validate_index(candidate_index)
                    if self.db.execute('SELECT 1 FROM evaluations WHERE campaign=? AND candidate_index=?',
                                       (campaign, candidate_index)).fetchone():
                        raise ValueError('Candidate was already evaluated.')
                    x = task.candidates[candidate_index].tolist()
                else:
                    x = request.get('x')
                bounds = self.status(campaign)['bounds']
                if not isinstance(x, list) or len(x) != len(bounds) or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    or not lo <= v <= hi for v, (lo, hi) in zip(x, bounds)):
                    raise ValueError('Expected finite coordinates within the public bounds.')
                label = self.db.execute('SELECT label FROM labels WHERE campaign=?', (campaign,)).fetchone()
                written = dict(self.db.execute('SELECT name,content FROM documents WHERE campaign=?',
                                               (campaign,)))
                if not label or not all(written.get(name, '').strip() for name in DOCUMENTS):
                    raise ValueError('Record the first decision and optimizer label before evaluating.')
                # Reserve a charged slot durably before calling the objective. A crash cannot reset it.
                self.db.execute('INSERT INTO evaluations(campaign,idx,request_id,x,label,wall_ns,candidate_index) '
                                'VALUES(?,?,?,?,?,?,?)',
                                (campaign, n + 1, key, json.dumps(x), label[0], time.time_ns(), candidate_index))
                self.db.commit()
                try:
                    ev = task.evaluate_index(candidate_index) if indexed else task.evaluate(x)
                except Exception as exc:
                    # Backend exceptions may identify the objective. Log the
                    # chained exception privately and expose only a generic error.
                    raise RuntimeError('Objective evaluation failed.') from exc
                result = float(ev.y)
                g = [float(v) for v in getattr(ev, 'g', ())]
                y_true = getattr(ev, 'y_true', None)
                if (not math.isfinite(result) or len(g) != self.status(campaign)['n_constraints']
                        or not all(math.isfinite(v) for v in g)):
                    raise RuntimeError('Nonfinite backend result')
                self.db.execute('UPDATE evaluations SET y=?,g=?,y_true=?,elapsed=? WHERE campaign=? AND idx=?',
                                (result, json.dumps(g), None if y_true is None else float(y_true),
                                 ev.elapsed, campaign, n + 1))
                if g:
                    result = {'y': result, 'g': g}
            else:
                raise ValueError('Unsupported evaluator operation.')
            reply = {'ok': True, 'result': result}
        except (ValueError, BudgetExhausted) as exc:
            reply = {'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)}
        except Exception:
            with self.error_log.open('a') as log:
                traceback.print_exc(file=log)
            reply = {'ok': False, 'error_type': 'EvaluatorError', 'error': 'Evaluator failed; stop and report.'}
        self.db.execute('UPDATE requests SET reply=? WHERE campaign=? AND id=?',
                        (json.dumps(reply, allow_nan=False), campaign, key))
        self.db.commit()
        return reply
