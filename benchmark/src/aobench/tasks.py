"""The benchmark's tasks: BBOB functions (via `ioh`) with optional COCO-style noise.

Every task minimizes on the box [-5, 5]^d. The seed picks the BBOB instance
(seed + 1, so a different shift and rotation) and the noise stream. Regret is
always scored on the noise-free value, which the optimizer never sees.

Noise is COCO's bbob-noisy multiplicative Gaussian model (Hansen et al. 2009,
RR-6869) applied to the gap g = f - f_opt: g * exp(beta * N(0, 1)). With
beta = 0.01 it is a 1% error; with beta = 1 it is as large as the gap itself.

Each task is there to show some solver failing (bad) next to some solver that
works (good). The evidence is the earlier axis studies (v4, v5: d = 5 and 20,
300·d evaluations, 15 seeds). Sobol is the reference: "bad" means doing no
better than Sobol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import ioh
import numpy as np

COCO_TOL = 1e-8
DIM = 5
BUDGET_PER_DIM = 300


@dataclass(frozen=True)
class Evaluation:
    y: float  # what the optimizer sees
    y_true: float  # noise-free value, for scoring only
    elapsed: float


class BBOBTask:
    def __init__(self, key: str, function_id: int, dim: int = DIM, beta: float = 0.0, seed: int = 0):
        self.name, self.function_id, self.dim, self.beta = key, function_id, dim, float(beta)
        self.bounds = np.array([[-5.0, 5.0]] * dim)
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._problem = ioh.get_problem(self.function_id, instance=self.seed + 1, dimension=self.dim,
                                        problem_class=ioh.ProblemClass.BBOB)
        self.f_opt = float(self._problem.optimum.y)
        self._noise = np.random.default_rng([self.seed, self.dim, self.function_id])

    def evaluate(self, x) -> Evaluation:
        x = np.asarray(x, dtype=float)
        start = time.perf_counter()
        y_true = float(self._problem(x))
        y = y_true
        g = y_true - self.f_opt
        if self.beta > 0 and g >= COCO_TOL:
            y = self.f_opt + g * float(np.exp(self.beta * self._noise.standard_normal())) + 1.01 * COCO_TOL
        return Evaluation(y, y_true, time.perf_counter() - start)


# key: (BBOB function id, noise beta, who is good, who is bad)
SPECS = {
    # f10 rotated ellipsoid, condition 1e6. Smooth and unimodal: gradients win.
    "ellipsoid": (10, 0.0, "lbfgsb", "sobol"),
    # The same ellipsoid with 1% noise: finite-difference gradients become noise.
    "noisy_ellipsoid": (10, 0.01, "cobyla, cma_es", "lbfgsb"),
    # f1 sphere with noise as large as the gap: COBYLA's linear models fit the noise.
    "very_noisy_sphere": (1, 1.0, "cma_es", "cobyla"),
    # f7 step ellipsoid: flat plateaus, zero gradient almost everywhere.
    "step_ellipsoid": (7, 0.0, "cma_es", "lbfgsb"),
    # f23 Katsuura: rugged at every scale, no large-scale trend to follow.
    "katsuura": (23, 0.0, "cobyla", "cma_es, turbo"),
}


def make_task(key: str, dim: int = DIM, seed: int = 0) -> BBOBTask:
    fid, beta, _, _ = SPECS[key]
    return BBOBTask(key, fid, dim=dim, beta=beta, seed=seed)
