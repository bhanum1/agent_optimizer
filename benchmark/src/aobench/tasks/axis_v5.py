"""Axis study v5: six baselines including TuRBO, on fresh BBOB instances (see benchmark/AXIS_V5_PLAN.md).

Six task groups, each aimed at one solver's failure mechanism. Five come from
v4 (three predicted there, two found post hoc); one is new and aimed at TuRBO:

- **plateau**: f7 step ellipsoid, d = 5, 20 (40 dropped, see DIMS), noiseless (Nelder–Mead);
- **noise_moderate**: f1 sphere, f9 rotated Rosenbrock, f10 ellipsoid with
  COCO's Gaussian noise at beta = 0.01 (L-BFGS-B);
- **noise_severe**: the same functions at beta = 1 (COBYLA);
- **katsuura**: f23 Katsuura, noiseless (CMA-ES; post hoc in v4);
- **weierstrass**: f16 Weierstrass, noiseless (CMA-ES at d >= 20; post hoc in v4);
- **illcond**: f10 ellipsoid, f11 discus, f12 bent cigar, noiseless (TuRBO).

As in v4, the functions are COCO's own (natively rotated and conditioned), and
noise is COCO's bbob-noisy Gaussian model (Hansen et al. 2009, RR-6869) on the
gap g = f - f_opt, reimplemented so it can be seeded. The seed picks the BBOB
instance (seed + 1) and the noise stream. v5 runs seeds 15..29, i.e. COCO
instances 16..30, none of which v4 (instances 1..15) used.
"""

from __future__ import annotations

import ioh
import numpy as np

from .base import Task

COCO_TOL = 1e-8
BUDGET_PER_DIM = 300
CHECKPOINTS_PER_DIM = (10, 30, 100, 300)
# d = 40 was planned but dropped for every solver and task before the sweep: TuRBO's local GP
# data keeps growing on illcond and plateau at d = 40, and the timing pilot put one 12,000-evaluation
# campaign at >= 19 h (see benchmark/AXIS_V5_PLAN.md). The budget stays 300*d.
DIMS = [5, 20]

GROUPS = {
    "plateau": dict(functions=[(7, "stepellipsoid")], betas=[0.0]),
    "noise_moderate": dict(functions=[(1, "sphere"), (9, "rosenbrockrot"), (10, "ellipsoid")], betas=[0.01]),
    "noise_severe": dict(functions=[(1, "sphere"), (9, "rosenbrockrot"), (10, "ellipsoid")], betas=[1.0]),
    "katsuura": dict(functions=[(23, "katsuura")], betas=[0.0]),
    "weierstrass": dict(functions=[(16, "weierstrass")], betas=[0.0]),
    "illcond": dict(functions=[(10, "ellipsoid"), (11, "discus"), (12, "bentcigar")], betas=[0.0]),
}


class AxisV5Task(Task):
    def __init__(self, function_id: int, fname: str, dim: int, beta: float, label: str,
                 seed: int = 15):
        self.function_id, self.fname, self.dim = function_id, fname, dim
        self.beta = float(beta)
        self.noisy = self.beta > 0
        self.name = label
        self.bounds = np.array([[-5.0, 5.0]] * dim)
        self.suggested_budget = BUDGET_PER_DIM * dim
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._problem = ioh.get_problem(self.function_id, instance=self.seed + 1,
                                        dimension=self.dim, problem_class=ioh.ProblemClass.BBOB)
        self.f_opt = float(self._problem.optimum.y)
        self.noise_rng = np.random.default_rng([self.seed, self.dim, self.function_id, 2026, 5])

    def _f_true(self, x) -> float:
        return float(self._problem(np.asarray(x, dtype=float)))

    def _f(self, x) -> float:
        y = self._f_true(x)
        if not self.noisy:
            return y
        g = y - self.f_opt
        if g >= COCO_TOL:
            g = g * float(np.exp(self.beta * self.noise_rng.standard_normal())) + 1.01 * COCO_TOL
        return self.f_opt + g


def task_key(group: str, fname: str, dim: int, beta: float) -> str:
    return f"v5_{group}_{fname}_d{dim}_b{beta:g}"


def axis_v5_cells(dims=DIMS) -> list[dict]:
    out = []
    for group, spec in GROUPS.items():
        for fid, fname in spec["functions"]:
            for dim in dims:
                for beta in spec["betas"]:
                    out.append(dict(key=task_key(group, fname, dim, beta), group=group,
                                    function_id=fid, fname=fname, dim=dim, beta=beta))
    return out


def axis_v5_tasks() -> dict:
    return {c["key"]: (lambda c=c: AxisV5Task(c["function_id"], c["fname"], c["dim"], c["beta"],
                                              label=c["key"]))
            for c in axis_v5_cells()}
