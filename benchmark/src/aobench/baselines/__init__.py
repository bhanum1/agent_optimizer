"""The baseline optimizers, and `make_solver` to build one by name.

Every baseline minimizes over a box and speaks the same ask/tell interface:

    opt = make_solver(name, bounds, seed, budget, noisy)
    X = opt.ask(1)          # a (1, d) array
    opt.tell(X, y)          # y: array of one observed value

Six are single methods; two (`shiwa`, `bandit`) select among methods. A solver
that has nothing more to propose raises OptimizerExhausted from `ask`.
"""
from __future__ import annotations


class OptimizerExhausted(Exception):
    """Raised by `ask` when the optimizer has nothing more to propose (a local solver that converged).
    Restarting catches it and starts again; the runner stops the campaign if it escapes."""


SINGLE = ["sobol", "lbfgsb", "cobyla", "cma_es", "smac", "gpbo"]
SELECTORS = ["shiwa", "bandit"]  # reported, but never set the "best" that soft fails are measured from
SOLVERS = SINGLE + SELECTORS
SLOW = {"gpbo", "smac", "bandit"}  # scheduled first, and logged one by one


def make_solver(name: str, bounds, seed: int, budget: int, noisy: bool = False):
    # Imported lazily: torch (GP-BO), SMAC and nevergrad (Shiwa) are slow to import.
    if name == "sobol":
        from .sobol import SobolSearch
        return SobolSearch(bounds, seed=seed)
    if name == "gpbo":
        from .gpbo import GPBO
        return GPBO(bounds, seed=seed)
    if name == "smac":
        from .smac import SMAC
        return SMAC(bounds, seed=seed, budget=budget)
    if name == "shiwa":
        from .shiwa import Shiwa
        return Shiwa(bounds, seed=seed, budget=budget, noisy=noisy)
    if name == "bandit":
        from .bandit import Bandit
        return Bandit(bounds, lambda arm: make_solver(arm, bounds, seed, budget, noisy), seed=seed, budget=budget)
    # The local solvers restart from a random point whenever they stop, so they use the whole budget.
    from .restarts import Restarting
    if name == "lbfgsb":
        from .lbfgsb import LBFGSB
        return Restarting(LBFGSB, bounds, seed)
    if name == "cobyla":
        from .cobyla import COBYLA
        return Restarting(COBYLA, bounds, seed)
    if name == "cma_es":
        from .cma_es import CMAES
        return Restarting(CMAES, bounds, seed, stop_when_converged=True)
    raise ValueError(f"unknown solver {name!r}; choose from {SOLVERS}")
