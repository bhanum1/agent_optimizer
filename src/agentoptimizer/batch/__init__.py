"""Batch proposers: from everything measured so far to the next q points.

Stateless: each call rebuilds what it needs from the data, so a proposal can
be reproduced from the inputs and seed alone. Used by advisor mode
(Campaign.inputs() and Campaign.write()); see base.py for the shared
signature and docs/plan/05-advisor-mode.md for the plan.
"""
from .base import Proposal, check_inputs, snap
from .space import random, space_filling
