"""The campaign's experiment table: data.csv.

One row per experiment. Bookkeeping columns come first, then one column per
parameter, the objective, and free-text notes. Other columns are the user's
and are kept as they are. Values are read as text; check_table() validates
them, and parse_table() turns a valid table into arrays.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
import os
from pathlib import Path

import numpy as np

STATUSES = ('done', 'pending', 'failed')
SOURCES = ('advisor', 'manual', 'import')
BOOKKEEPING = ('id', 'block', 'status', 'source', 'optimizer')


class DataError(ValueError):
    """data.csv cannot be read or used as it is; the message says where."""


@dataclass(frozen=True)
class Issue:
    level: str  # 'error' or 'warning'
    where: str
    message: str

    def __str__(self):
        return f'{self.level}: {self.where}: {self.message}'


@dataclass
class Table:
    columns: list[str]
    rows: list[dict] = field(default_factory=list)
    # File line of each row, for messages; None for rows not yet written.
    lines: list = field(default_factory=list)


@dataclass(frozen=True)
class Data:
    """A valid table as arrays, in file order. X holds raw parameter values;
    y holds objective values in the user's direction, NaN unless done."""
    ids: np.ndarray
    blocks: np.ndarray
    status: np.ndarray
    source: np.ndarray
    optimizer: np.ndarray
    X: np.ndarray
    y: np.ndarray

    def where(self, status):
        return self.status == status


def standard_columns(problem):
    return [*BOOKKEEPING, *problem.names, problem.objective, 'notes']


def required_columns(problem):
    return [*BOOKKEEPING, *problem.names, problem.objective]


def read_table(folder, problem):
    path = Path(folder) / 'data.csv'
    if not path.exists():
        return Table(standard_columns(problem))
    with path.open(newline='') as stream:
        reader = csv.reader(stream)
        header = [cell.strip() for cell in next(reader, [])]
        if not header:
            return Table(standard_columns(problem))
        empty = [i + 1 for i, name in enumerate(header) if not name]
        if empty:
            raise DataError(f'{path}: header cell(s) {empty} are empty')
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise DataError(f'{path}: duplicate column(s) {", ".join(duplicates)}')
        table = Table(header)
        for line in reader:
            if not any(cell.strip() for cell in line):
                continue
            if len(line) > len(header):
                raise DataError(f'{path}, line {reader.line_num}: {len(line)} cells, '
                                f'but the header has {len(header)}')
            cells = [cell.strip() for cell in line] + [''] * (len(header) - len(line))
            table.rows.append(dict(zip(header, cells)))
            table.lines.append(reader.line_num)
    return table


def write_table(folder, table):
    path = Path(folder) / 'data.csv'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=table.columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({column: row.get(column, '') for column in table.columns} for row in table.rows)
    os.replace(temporary, path)


def format_number(parameter, value):
    """Text for a parameter value: whole numbers for integers, shortest exact float otherwise."""
    return str(int(round(value))) if parameter.type == 'integer' else repr(float(value))


def _float(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _int(text):
    try:
        return int(text)
    except ValueError:
        return None


def check_table(problem, table):
    """Every problem with the table, errors first in file order, then warnings."""
    issues = []
    missing = [c for c in required_columns(problem) if c not in table.columns]
    if missing:
        return [Issue('error', 'data.csv header', 'missing column(s) ' + ', '.join(missing)
                      + '; expected ' + ', '.join(standard_columns(problem)))]

    def where(row, line):
        place = f'line {line}' if line else 'new row'
        return place + (f' (id {row["id"]})' if _int(row['id']) is not None else '')

    def error(row, line, column, message):
        issues.append(Issue('error', f'{where(row, line)}, {column}', message))

    seen, points = {}, {}
    lines = table.lines + [None] * (len(table.rows) - len(table.lines))
    for row, line in zip(table.rows, lines):
        identifier = _int(row['id'])
        if identifier is None or identifier < 1:
            error(row, line, 'id', f'must be a positive whole number, not {row["id"]!r}')
        elif identifier in seen:
            error(row, line, 'id', f'repeats the id on line {seen[identifier]}')
        else:
            seen[identifier] = line
        block = _int(row['block'])
        if block is None or block < 0:
            error(row, line, 'block', f'must be a whole number >= 0, not {row["block"]!r}')
        status = row['status']
        if status not in STATUSES:
            error(row, line, 'status', f'must be one of {", ".join(STATUSES)}, not {status!r}')
        if row['source'] not in SOURCES:
            error(row, line, 'source', f'must be one of {", ".join(SOURCES)}, not {row["source"]!r}')
        if not row['optimizer']:
            error(row, line, 'optimizer', 'is empty; use the method label, or "manual" or "import"')

        point = []
        for p in problem.parameters:
            value = _float(row[p.name])
            if value is None:
                error(row, line, p.name, f'must be a finite number, not {row[p.name]!r}')
            elif not p.low <= value <= p.high:
                error(row, line, p.name, f'{value:g} is outside [{p.low:g}, {p.high:g}]; '
                      'fix the row or widen the bounds in problem.toml')
            elif p.type == 'integer' and not value.is_integer():
                error(row, line, p.name, f'{value:g} is not a whole number')
            point.append(value)

        result = row[problem.objective]
        if status == 'done' and _float(result) is None:
            error(row, line, problem.objective, f'a done row needs a finite value, not {result!r}')
        elif status == 'pending' and result:
            error(row, line, problem.objective, 'has a result but status pending; '
                  'set status to done once the experiment has run')
        elif status == 'failed' and result:
            error(row, line, problem.objective, 'a failed row has no result; clear it, or set status to done')

        if status == 'pending' and None not in point:
            key = tuple(point)
            if key in points:
                issues.append(Issue('warning', where(row, line),
                                    f'repeats the point of id {points[key]}, which is also pending or done'))
        if status in ('pending', 'done') and None not in point:
            points.setdefault(tuple(point), row['id'])
    return sorted(issues, key=lambda issue: issue.level != 'error')


def parse_table(problem, table):
    errors = [issue for issue in check_table(problem, table) if issue.level == 'error']
    if errors:
        raise DataError(f'data.csv has {len(errors)} error(s); run: opt_run.py check <campaign>\n'
                        + '\n'.join(str(issue) for issue in errors[:10]))
    rows = table.rows
    X = np.array([[float(row[name]) for name in problem.names] for row in rows]).reshape(len(rows), problem.dim)
    y = np.array([float(row[problem.objective]) if row['status'] == 'done' else np.nan for row in rows])
    text = lambda column: np.array([row[column] for row in rows], dtype=object)
    return Data(ids=np.array([int(row['id']) for row in rows], dtype=int),
                blocks=np.array([int(row['block']) for row in rows], dtype=int),
                status=text('status'), source=text('source'), optimizer=text('optimizer'), X=X, y=y)
