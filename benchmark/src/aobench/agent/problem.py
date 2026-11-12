"""The starting files of a campaign: a task's problem definition, with its identity withheld."""
from __future__ import annotations


def problem_yaml(task) -> str:
    spec = task.spec
    noisy = str(spec.beta > 0).lower()
    lines = ["name: problem",  # never the task's name: the agent must not know which function this is
             "objective: {name: y, direction: minimize}",
             "variables:"]
    lines += [f"  - {{name: x{i}, type: continuous, low: -5, high: 5}}" for i in range(spec.dim)]
    lines += [f"noise: {{present: {noisy}, replicates_allowed: {noisy}}}",
              f"budget: {spec.budget}", "parallel: 1", f"seed: {task.seed}"]
    return "\n".join(lines) + "\n"


def problem_md(task) -> str:
    spec = task.spec
    noise = ("Repeated measurements of the same point differ: the objective is noisy, by an unknown amount. "
             "Replicates are allowed." if spec.beta > 0 else
             "The objective is deterministic: the same point always gives the same value.")
    return f"""# Problem

Minimize `y`, a black-box function of {spec.dim} continuous variables `x0`..`x{spec.dim - 1}`, each in [-5, 5].

The objective's identity is withheld: there is no domain knowledge about it.

{noise}

Each measurement is cheap and quick, and is made one at a time. The budget is {spec.budget} measurements, with \
no earlier data. The result is the lowest `y` measured.
"""
