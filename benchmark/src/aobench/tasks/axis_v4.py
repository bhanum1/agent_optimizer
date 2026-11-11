"""Axis study v4: rank-reversal tasks for five baselines (see benchmark/AXIS_V4_PLAN.md).

Four task groups, each built around a failure mechanism documented in the
literature, so that a pair of solvers swaps order between groups:

- **illcond**: COCO's ill-conditioned group, f10 ellipsoid, f11 discus,
  f12 bent cigar (conditioning 1e6, rotated), d = 5, 20, 40, noiseless;
- **noise**: f1 sphere, f9 rotated Rosenbrock, f10 ellipsoid with COCO's
  Gaussian noise model at beta = 0.01 (moderate) and 1 (severe), d = 5, 20, 40;
- **highdim**: f1, f9, f10 at d = 60 and 100, noiseless;
- **rugged**: f16 Weierstrass, f18 Schaffer F7 (conditioning 1e3),
  f23 Katsuura, d = 5, 20, 40, noiseless. None of these were in v3, so this
  group tests v3's post-hoc "rugged" finding on new functions;
- **plateau** (amendment 1): f7 step ellipsoid, d = 5, 20, 40, noiseless. Its
  plateaus stall CMA-ES's step size and zero finite-difference gradients.

Unlike v3, the functions are COCO's own, with no extra transform: COCO already
rotates and conditions them. Noise is COCO's bbob-noisy Gaussian model
(Hansen et al. 2009, RR-6869) on the gap g = f - f_opt, reimplemented so it can
be seeded (as in v3). The seed picks the BBOB instance (seed + 1, so seeds
0..14 are COCO's instances 1..15) and the noise stream.
"""

from __future__ import annotations

import ioh
import numpy as np

from .base import Task

COCO_TOL = 1e-8
BUDGET_PER_DIM = 300
CHECKPOINTS_PER_DIM = (10, 30, 100, 300)

GROUPS = {
    "illcond": dict(functions=[(10, "ellipsoid"), (11, "discus"), (12, "bentcigar")],
                    dims=[5, 20, 40], betas=[0.0]),
    "noise": dict(functions=[(1, "sphere"), (9, "rosenbrockrot"), (10, "ellipsoid")],
                  dims=[5, 20, 40], betas=[0.01, 1.0]),
    "highdim": dict(functions=[(1, "sphere"), (9, "rosenbrockrot"), (10, "ellipsoid")],
                    dims=[60, 100], betas=[0.0]),
    "rugged": dict(functions=[(16, "weierstrass"), (18, "schafferf7cond"), (23, "katsuura")],
                   dims=[5, 20, 40], betas=[0.0]),
    # Added 2026-11-10 (amendment 1), run separately on a workstation.
    "plateau": dict(functions=[(7, "stepellipsoid")], dims=[5, 20, 40], betas=[0.0]),
}


class AxisV4Task(Task):
    def __init__(self, function_id: int, fname: str, dim: int, beta: float, label: str,
                 seed: int = 0):
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
        self.noise_rng = np.random.default_rng([self.seed, self.dim, self.function_id, 2026, 4])

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
    return f"v4_{group}_{fname}_d{dim}_b{beta:g}"


def axis_v4_cells() -> list[dict]:
    out = []
    for group, spec in GROUPS.items():
        for fid, fname in spec["functions"]:
            for dim in spec["dims"]:
                for beta in spec["betas"]:
                    out.append(dict(key=task_key(group, fname, dim, beta), group=group,
                                    function_id=fid, fname=fname, dim=dim, beta=beta))
    return out


def axis_v4_tasks() -> dict:
    return {c["key"]: (lambda c=c: AxisV4Task(c["function_id"], c["fname"], c["dim"], c["beta"],
                                              label=c["key"]))
            for c in axis_v4_cells()}
