"""The benchmark's tasks: BBOB functions (via `ioh`), some with noise.

Every solver sees the box [-5, 5]^d and minimizes. The seed picks the BBOB
instance (seed + 1, so a different shift and rotation) and the noise stream.
Regret is always scored on the noise-free value, which the optimizer never sees.

**Noise** is COCO's bbob-noisy multiplicative Gaussian model (Hansen et al.
2009, RR-6869) on the gap g = f - f_opt: g * exp(beta * N(0, 1)). beta = 0.01 is
a 1% error; beta = 1 is as large as the gap itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import ioh
import numpy as np

COCO_TOL = 1e-8


@dataclass(frozen=True)
class Evaluation:
    y: float  # what the optimizer sees
    y_true: float  # noise-free value, for scoring only
    elapsed: float


@dataclass(frozen=True)
class Spec:
    function_id: int
    dim: int
    budget: int
    good: str  # who does well, for the report
    bad: str
    beta: float = 0.0  # noise


class BBOBTask:
    def __init__(self, key: str, spec: Spec, seed: int = 0):
        self.name, self.spec, self.dim = key, spec, spec.dim
        self.function_id, self.beta = spec.function_id, float(spec.beta)
        self.bounds = np.array([[-5.0, 5.0]] * self.dim)
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._problem = ioh.get_problem(self.function_id, instance=self.seed + 1, dimension=self.dim,
                                        problem_class=ioh.ProblemClass.BBOB)
        self.f_opt = float(self._problem.optimum.y)
        self._noise = np.random.default_rng([self.seed, self.dim, self.function_id])

    def evaluate(self, x) -> Evaluation:
        start = time.perf_counter()
        y_true = float(self._problem(np.asarray(x, dtype=float)))
        y = y_true
        g = y_true - self.f_opt
        if self.beta > 0 and g >= COCO_TOL:
            y = self.f_opt + g * float(np.exp(self.beta * self._noise.standard_normal())) + 1.01 * COCO_TOL
        return Evaluation(y, y_true, time.perf_counter() - start)


SPECS = {
    # f10 rotated ellipsoid, condition 1e6. Smooth and unimodal: gradients win.
    "ellipsoid": Spec(10, 5, 1500, "lbfgsb", "cobyla, smac"),
    # The same ellipsoid with 1% noise: finite-difference gradients become noise.
    "noisy_ellipsoid": Spec(10, 5, 1500, "cma_es", "lbfgsb", beta=0.01),
    # f1 sphere with noise as large as the gap: L-BFGS-B's finite differences and
    # COBYLA's linear models fit the noise.
    "very_noisy_sphere": Spec(1, 5, 1500, "smac", "lbfgsb, cobyla", beta=1.0),
    # f7 step ellipsoid: flat plateaus, zero gradient almost everywhere.
    "step_ellipsoid": Spec(7, 5, 1500, "cma_es", "lbfgsb"),
    # f1 sphere with 1% noise at d = 10 and 30d evaluations: COBYLA's linear models converge
    # fast through the noise; finite differences turn to noise; CMA-ES is too slow to start.
    "noisy_sphere": Spec(1, 10, 300, "cobyla", "lbfgsb, cma_es, smac", beta=0.01),
    # f21 Gallagher, 101 Gaussian peaks, at d = 2 and 30d evaluations: a global GP finds the
    # best peak; local solvers climb whichever peak is nearest.
    "gallagher": Spec(21, 2, 60, "gpbo", "lbfgsb, cobyla, cma_es"),
}


def make_task(key: str, seed: int = 0) -> BBOBTask:
    return BBOBTask(key, SPECS[key], seed=seed)
