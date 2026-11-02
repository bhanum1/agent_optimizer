"""The campaign's problem definition: problem.toml.

The user states the parameters, the objective and the budget here; every
other part of advisor mode reads them through load_problem(). Unknown keys
are errors, so a typo cannot silently drop a setting.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import tomllib

import numpy as np

# Columns of data.csv with a fixed meaning; parameters and the objective may
# not use these names.
RESERVED = ('id', 'block', 'status', 'source', 'optimizer', 'candidate_id', 'notes')
TYPES = ('continuous', 'integer')
DIRECTIONS = ('minimize', 'maximize')
NOISE = ('unknown', 'none', 'low', 'high')

TEMPLATE = '''\
# Advisor-mode campaign. Edit, then run: opt_run.py check <this folder>
name = "my campaign"
description = """
What is being optimized and anything an expert would want to know:
the process, known trends, failure modes, measurement noise.
"""
noise = "unknown"            # unknown | none | low | high

[objective]
name = "yield"               # the column in data.csv that holds the result
direction = "maximize"       # maximize | minimize

[budget]
total = 40                   # experiments you can afford in all (optional)
block_size = 8               # experiments you prefer to run per block (optional)

# One [[parameters]] table per input you control.
[[parameters]]
name = "temperature"
type = "continuous"          # continuous | integer
low = 80
high = 200
unit = "C"

[[parameters]]
name = "concentration"
type = "continuous"
low = 0.001
high = 1.0
log = true                   # search on a log scale (continuous, low > 0)
unit = "M"

[[parameters]]
name = "stir_minutes"
type = "integer"
low = 1
high = 30
unit = "min"
'''


class ProblemError(ValueError):
    """problem.toml is missing, malformed or inconsistent; the message says where."""


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str
    low: float
    high: float
    log: bool = False
    unit: str = ''

    def to_unit(self, values):
        values = np.asarray(values, dtype=float)
        if self.log:
            return (np.log(values) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        if self.type == 'integer':
            # Each integer owns an equal slice of [0, 1].
            return (values - self.low + .5) / (self.high - self.low + 1)
        return (values - self.low) / (self.high - self.low)

    def from_unit(self, units):
        units = np.clip(np.asarray(units, dtype=float), 0., 1.)
        if self.log:
            values = np.exp(math.log(self.low) + units * (math.log(self.high) - math.log(self.low)))
        elif self.type == 'integer':
            values = np.floor(self.low + units * (self.high - self.low + 1))
        else:
            values = self.low + units * (self.high - self.low)
        return np.clip(values, self.low, self.high)


@dataclass(frozen=True, eq=False)
class Pool:
    """A finite set of candidates: experiments can only be chosen from these rows."""
    file: str
    ids: tuple
    names: list  # parameter names, the columns of X
    X: np.ndarray  # raw feature values

    def __len__(self):
        return len(self.ids)

    def rows(self):
        """candidate_id -> row position."""
        return {identifier: i for i, identifier in enumerate(self.ids)}


@dataclass(frozen=True)
class Problem:
    name: str
    description: str
    noise: str
    objective: str
    direction: str
    parameters: tuple[Parameter, ...]
    total_budget: int | None = None
    block_size: int | None = None
    pool: Pool | None = field(default=None, compare=False)

    @property
    def names(self):
        return [p.name for p in self.parameters]

    @property
    def dim(self):
        return len(self.parameters)

    @property
    def bounds(self):
        return np.array([[p.low, p.high] for p in self.parameters])

    @property
    def sign(self):
        """Multiply objective values by this to get a quantity to minimize."""
        return 1. if self.direction == 'minimize' else -1.

    def to_unit(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.column_stack([p.to_unit(X[:, i]) for i, p in enumerate(self.parameters)]) \
            if len(X) else np.empty((0, self.dim))

    def from_unit(self, U):
        U = np.atleast_2d(np.asarray(U, dtype=float))
        return np.column_stack([p.from_unit(U[:, i]) for i, p in enumerate(self.parameters)]) \
            if len(U) else np.empty((0, self.dim))


def _keys(table, allowed, where):
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ProblemError(f'{where}: unknown key(s) {", ".join(unknown)}; allowed: {", ".join(allowed)}')


def _number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProblemError(f'{where} must be a finite number, not {value!r}')
    return float(value)


def _count(value, where):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProblemError(f'{where} must be a positive integer, not {value!r}')
    return value


def _parameter(table, index, pool=None):
    where = f'parameters[{index}]'
    if not isinstance(table, dict):
        raise ProblemError(f'{where} must be a [[parameters]] table')
    _keys(table, ('name', 'type', 'low', 'high', 'log', 'unit'), where)
    name = table.get('name')
    if not isinstance(name, str) or not name.strip() or name != name.strip():
        raise ProblemError(f'{where}.name must be a non-empty string without surrounding spaces')
    where = f'parameter "{name}"'
    kind = table.get('type', 'continuous')
    if kind not in TYPES:
        raise ProblemError(f'{where}: type must be one of {", ".join(TYPES)}, not {kind!r}')
    column = pool.X[:, pool.names.index(name)] if pool is not None else None
    for key in ('low', 'high'):
        if key not in table and column is None:
            raise ProblemError(f'{where}: {key} is required')
    # With a pool, missing bounds are the range of its candidates; a constant
    # feature gets a unit-wide range so scaling stays defined.
    low = _number(table['low'], f'{where}.low') if 'low' in table else float(column.min())
    high = _number(table['high'], f'{where}.high') if 'high' in table else float(column.max())
    if column is not None and 'high' not in table and high == low:
        high = low + 1.
    if column is not None and (column.min() < low or column.max() > high):
        raise ProblemError(f'{where}: candidates span [{column.min():g}, {column.max():g}], '
                           f'outside the bounds [{low:g}, {high:g}]')
    if column is not None and kind == 'integer' and not np.all(column == np.round(column)):
        raise ProblemError(f'{where}: an integer parameter has non-integer candidate values')
    if not low < high:
        raise ProblemError(f'{where}: low ({low:g}) must be below high ({high:g})')
    log = table.get('log', False)
    if not isinstance(log, bool):
        raise ProblemError(f'{where}: log must be true or false')
    if kind == 'integer' and not (low.is_integer() and high.is_integer()):
        raise ProblemError(f'{where}: an integer parameter needs whole-number bounds')
    if log and kind != 'continuous':
        raise ProblemError(f'{where}: log scaling is for continuous parameters')
    if log and low <= 0:
        raise ProblemError(f'{where}: log scaling needs low > 0')
    unit = table.get('unit', '')
    if not isinstance(unit, str):
        raise ProblemError(f'{where}: unit must be a string')
    return Parameter(name, kind, low, high, log, unit)


def read_pool(path, names):
    """Read candidates.csv: a candidate_id column and one column per parameter."""
    path = Path(path)
    if not path.exists():
        raise ProblemError(f'[pool] file {path} not found')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        columns = [c.strip() for c in reader.fieldnames or []]
        missing = [c for c in ['candidate_id', *names] if c not in columns]
        if missing:
            raise ProblemError(f'{path}: missing column(s) {", ".join(missing)}')
        ids, X = [], []
        for row in reader:
            row = {k.strip(): (v or '').strip() for k, v in row.items() if k is not None}
            if not any(row.values()):
                continue
            where = f'{path}, line {reader.line_num}'
            if not row['candidate_id']:
                raise ProblemError(f'{where}: candidate_id is empty')
            values = []
            for name in names:
                try:
                    value = float(row[name])
                except ValueError:
                    value = math.nan
                if not math.isfinite(value):
                    raise ProblemError(f'{where}: {name} must be a finite number, not {row[name]!r}')
                values.append(value)
            ids.append(row['candidate_id'])
            X.append(values)
    duplicates = sorted(i for i, n in Counter(ids).items() if n > 1)
    if duplicates:
        raise ProblemError(f'{path}: repeated candidate_id(s) {", ".join(duplicates[:10])}')
    if not ids:
        raise ProblemError(f'{path} has no candidates')
    return Pool(path.name, tuple(ids), list(names), np.array(X, dtype=float))


def parse_problem(data, default_name='campaign', folder=None):
    _keys(data, ('name', 'description', 'noise', 'objective', 'budget', 'parameters', 'pool'), 'problem.toml')
    name = data.get('name', default_name)
    description = data.get('description', '')
    if not isinstance(name, str) or not isinstance(description, str):
        raise ProblemError('name and description must be strings')
    noise = data.get('noise', 'unknown')
    if noise not in NOISE:
        raise ProblemError(f'noise must be one of {", ".join(NOISE)}, not {noise!r}')

    objective = data.get('objective')
    if not isinstance(objective, dict):
        raise ProblemError('[objective] is required, with name and direction')
    _keys(objective, ('name', 'direction'), '[objective]')
    objective_name = objective.get('name')
    if not isinstance(objective_name, str) or not objective_name.strip():
        raise ProblemError('[objective] name must be a non-empty string')
    direction = objective.get('direction')
    if direction not in DIRECTIONS:
        raise ProblemError(f'[objective] direction must be one of {", ".join(DIRECTIONS)}, not {direction!r}')

    budget = data.get('budget', {})
    if not isinstance(budget, dict):
        raise ProblemError('[budget] must be a table')
    _keys(budget, ('total', 'block_size'), '[budget]')
    total = _count(budget['total'], '[budget] total') if 'total' in budget else None
    block = _count(budget['block_size'], '[budget] block_size') if 'block_size' in budget else None
    if total and block and block > total:
        raise ProblemError(f'[budget] block_size ({block}) exceeds total ({total})')

    tables = data.get('parameters')
    if not isinstance(tables, list) or not tables:
        raise ProblemError('at least one [[parameters]] table is required')
    pool = None
    if 'pool' in data:
        spec = data['pool']
        if not isinstance(spec, dict):
            raise ProblemError('[pool] must be a table')
        _keys(spec, ('file',), '[pool]')
        if folder is None:
            raise ProblemError('[pool] is read from the campaign folder; use load_problem()')
        names = [t.get('name') if isinstance(t, dict) else None for t in tables]
        if not all(isinstance(n, str) and n for n in names):
            raise ProblemError('every [[parameters]] table needs a name')
        pool = read_pool(Path(folder) / spec.get('file', 'candidates.csv'), names)
    parameters = tuple(_parameter(table, i, pool) for i, table in enumerate(tables))
    names = [p.name for p in parameters] + [objective_name]
    for column in names:
        if column in RESERVED:
            raise ProblemError(f'"{column}" is a reserved column name; reserved: {", ".join(RESERVED)}')
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ProblemError(f'names must be unique across parameters and the objective: {", ".join(duplicates)}')
    if pool is not None and total and total > len(pool):
        raise ProblemError(f'[budget] total ({total}) exceeds the {len(pool)} candidates in the pool')
    return Problem(name, description.strip(), noise, objective_name, direction, parameters, total, block, pool)


def load_problem(folder):
    path = Path(folder) / 'problem.toml'
    if not path.exists():
        raise ProblemError(f'{path} not found; create one with: opt_run.py init {folder}')
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ProblemError(f'{path}: {exc}') from None
    return parse_problem(data, default_name=Path(folder).resolve().name, folder=folder)
