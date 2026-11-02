"""Advisor mode: propose the next block of a real campaign from its data so far.

See docs/plan/05-advisor-mode.md. A campaign is a folder holding problem.toml,
data.csv, decisions.md and one blocks/NNN/ folder per proposed block.
"""
from .campaign import Campaign
from .data import Data, DataError, Issue, Table, check_table, parse_table, read_table, write_table
from .problem import Parameter, Problem, ProblemError, load_problem, parse_problem
