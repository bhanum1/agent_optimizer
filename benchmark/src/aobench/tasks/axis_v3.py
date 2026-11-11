"""Axis study v3: BBOB shapes crossed with conditioning, dimension and noise.

A full factorial (see benchmark/AXIS_STUDY_PLAN.md):

- **global structure** (the function, three BBOB shapes per level):
  unimodal f1 sphere, f8 Rosenbrock, f13 sharp ridge; multimodal with adequate
  global structure f15 Rastrigin, f17 Schaffer F7, f19 Griewank-Rosenbrock;
  multimodal with weak global structure f20 Schwefel, f21 Gallagher 101,
  f24 Lunacek bi-Rastrigin;
- **conditioning** 1, 1e3, 1e6, added about the optimum as a rotated scaling
  T = Q diag(lam) R (Q, R random orthogonal, lam from 1 down to 1/sqrt(kappa),
  log-spaced, so curvature ratios grow by kappa). At kappa = 1 T is a pure
  rotation, so every shape is non-separable;
- **dimension** 5, 20, 40;
- **noise**: COCO's bbob-noisy Gaussian model (Hansen et al. 2009, RR-6869),
  applied to the gap g = f - f_opt: g * exp(beta * N(0, 1)) + 1.01e-8 when
  g >= 1e-8, else g unchanged. beta = 0, 0.01 (moderate), 1 (severe). COCO's own
  noise generator cannot be seeded, so the formula is reimplemented here.

All coordinates are continuous and active, in [-5, 5]. The seed picks the BBOB
instance (seed + 1), the rotation and the noise stream; f_opt stays exact
because T fixes x*.
"""

from __future__ import annotations

import ioh
import numpy as np

from .base import Task

STRUCTURE = {
    "unimodal": [(1, "sphere"), (8, "rosenbrock"), (13, "sharpridge")],
    "adequate": [(15, "rastrigin"), (17, "schafferf7"), (19, "griewankrosenbrock")],
    "weak": [(20, "schwefel"), (21, "gallagher101"), (24, "lunacek")],
}
CONDITIONING = {"low": 1.0, "medium": 1e3, "high": 1e6}
DIMENSION = {"low": 5, "medium": 20, "high": 40}
NOISE = {"low": 0.0, "medium": 0.01, "high": 1.0}  # COCO's beta
COCO_TOL = 1e-8
BUDGET_PER_DIM = 300
CHECKPOINTS_PER_DIM = (10, 30, 100, 300)


class AxisV3Task(Task):
    def __init__(self, function_id: int, fname: str, dim: int, kappa: float, beta: float,
                 label: str, seed: int = 0):
        self.function_id, self.fname, self.dim = function_id, fname, dim
        self.kappa, self.beta = float(kappa), float(beta)
        self.noisy = self.beta > 0
        self.name = label
        self.bounds = np.array([[-5.0, 5.0]] * dim)
        self.suggested_budget = BUDGET_PER_DIM * dim
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        d = self.dim
        self._problem = ioh.get_problem(self.function_id, instance=self.seed + 1, dimension=d,
                                        problem_class=ioh.ProblemClass.BBOB)
        self.f_opt = float(self._problem.optimum.y)
        self.x_star = np.asarray(self._problem.optimum.x, dtype=float)
        rng = np.random.default_rng([self.function_id, d, self.seed, 20261110])
        q, r = (self._orthogonal(rng, d) for _ in range(2))
        lam = self.kappa ** (-0.5 * np.arange(d) / max(d - 1, 1))
        self.transform = q @ np.diag(lam) @ r
        self.noise_rng = np.random.default_rng([self.seed, d, self.function_id, 77])

    @staticmethod
    def _orthogonal(rng, d) -> np.ndarray:
        q, r = np.linalg.qr(rng.standard_normal((d, d)))
        return q * np.sign(np.diag(r))

    def to_base(self, x) -> np.ndarray:
        return self.x_star + self.transform @ (np.asarray(x, dtype=float) - self.x_star)

    def _f_true(self, x) -> float:
        return float(self._problem(self.to_base(x)))

    def _f(self, x) -> float:
        y = self._f_true(x)
        if not self.noisy:
            return y
        g = y - self.f_opt
        if g >= COCO_TOL:
            g = g * float(np.exp(self.beta * self.noise_rng.standard_normal())) + 1.01 * COCO_TOL
        return self.f_opt + g


def task_key(fname: str, dim: int, kappa: float, beta: float) -> str:
    return f"v3_{fname}_d{dim}_k{kappa:g}_b{beta:g}"


def axis_v3_cells() -> list[dict]:
    """Every (function, dimension, conditioning, noise) with its axis levels."""
    out = []
    for s_level, fns in STRUCTURE.items():
        for fid, fname in fns:
            for c_level, kappa in CONDITIONING.items():
                for d_level, dim in DIMENSION.items():
                    for n_level, beta in NOISE.items():
                        out.append(dict(key=task_key(fname, dim, kappa, beta), function_id=fid,
                                        fname=fname, dim=dim, kappa=kappa, beta=beta,
                                        structure=s_level, conditioning=c_level,
                                        dimension=d_level, noise=n_level))
    return out


def axis_v3_tasks() -> dict:
    """Registry entries: key -> factory."""
    return {c["key"]: (lambda c=c: AxisV3Task(c["function_id"], c["fname"], c["dim"], c["kappa"],
                                              c["beta"], label=c["key"]))
            for c in axis_v3_cells()}
