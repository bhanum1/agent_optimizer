"""Synthetic tasks for the axis study: one BBOB function plus wrappers.

Each task is a noiseless BBOB function from `ioh` with five properties set
independently (the design is in `L18` below):

- **modality** (by choice of function): unimodal, multimodal with strong global
  structure, multimodal with weak global structure (BBOB's function groups);
- **conditioning**: extra ill-conditioning applied about the optimum;
- **dimension**;
- **noise**: additive Gaussian noise with a fixed sigma, like a lab measurement:
  y = f + sigma * N(0, 1). sigma is a multiple of a "good point" value: the gap
  f - f_opt of a point better than 90% of uniformly random points (the 10th
  percentile of the gap over 2,000 such points). Moderate is 0.1x, severe 1x;
- **variable type**: all continuous, or 80% of the coordinates integer or
  categorical (the `bbob-mixint` recipe);
- **effective dimension**: the fraction of the coordinates that affect the value
  (on mixed tasks at least one active coordinate is continuous).

Which coordinates are active and which are discrete (and their level counts) is
fixed per task: it depends only on the task's settings, never on the seed. The
seed (via `reseed`) picks the BBOB instance, the category permutations, the
conditioning rotation and the noise stream, so seeds are replicates of one task. `f_opt` stays
exact: every wrapper keeps the BBOB optimum, shifted onto the grid for discrete
coordinates.

Public point (what optimizers and the agent see): continuous coordinates in
[-5, 5]; a discrete coordinate with k levels is its level index 0..k-1 (bounds
[-0.5, k-0.5], rounded on evaluation). For a categorical coordinate the index is
a label's position, and a hidden permutation maps labels to levels, so the order
of the labels carries no information.
"""

from __future__ import annotations

import ioh
import numpy as np

from .base import Task

# (BBOB id, name) per modality level; two functions per level.
FUNCTIONS = {
    "unimodal": [(1, "sphere"), (9, "rosenbrock")],
    "strong": [(15, "rastrigin"), (17, "schaffer")],
    "weak": [(21, "gallagher101"), (20, "schwefel")],
}
CONDITIONING = {"low": 1.0, "medium": 1e3, "high": 1e6}  # added on top of the function's own
DIMENSION = {"low": 5, "medium": 10, "high": 20}
NOISE = {"low": 0.0, "medium": 0.1, "high": 1.0}  # sigma as a multiple of the good-point gap
GOOD_POINT_QUANTILE = 0.10  # good point: better than 90% of uniformly random points
NOISE_SAMPLES = 2000
VARTYPE = {"low": "continuous", "medium": "integer", "high": "categorical"}
ACTIVE = {"low": 1.0, "medium": 0.5, "high": 0.2}  # fraction of coordinates that matter
ARITIES = (2, 4, 8, 16)  # bbob-mixint's level counts, cycled over the discrete coordinates
DISCRETE_SHARE = 0.8  # bbob-mixint: 4 of every 5 coordinates are discrete

# Taguchi L18, OA(18, 2^1 3^7, strength 2). Column 0: which of the level's two
# functions; columns 1-6: modality, conditioning, dimension, noise, variable type,
# effective dimension; column 7 is unused.
L18 = [
    [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 2, 2, 2, 2, 2, 2], [1, 1, 3, 3, 3, 3, 3, 3],
    [1, 2, 1, 1, 2, 2, 3, 3], [1, 2, 2, 2, 3, 3, 1, 1], [1, 2, 3, 3, 1, 1, 2, 2],
    [1, 3, 1, 2, 1, 3, 2, 3], [1, 3, 2, 3, 2, 1, 3, 1], [1, 3, 3, 1, 3, 2, 1, 2],
    [2, 1, 1, 3, 3, 2, 2, 1], [2, 1, 2, 1, 1, 3, 3, 2], [2, 1, 3, 2, 2, 1, 1, 3],
    [2, 2, 1, 2, 3, 1, 3, 2], [2, 2, 2, 3, 1, 2, 1, 3], [2, 2, 3, 1, 2, 3, 2, 1],
    [2, 3, 1, 3, 2, 3, 1, 2], [2, 3, 2, 1, 3, 1, 2, 3], [2, 3, 3, 2, 1, 2, 3, 1],
]
LEVELS = {1: "low", 2: "medium", 3: "high"}
MODALITY = {1: "unimodal", 2: "strong", 3: "weak"}


class SyntheticTask(Task):
    def __init__(self, function_id: int, fname: str, dim: int, conditioning: float = 1.0,
                 noise: float = 0.0, vartype: str = "continuous", active: float = 1.0,
                 label: str | None = None, seed: int = 0):
        self.function_id, self.fname = function_id, fname
        self.dim, self.kappa, self.noise_level = dim, float(conditioning), float(noise)
        self.vartype, self.active_share = vartype, float(active)
        self.noisy = self.noise_level > 0
        self.suggested_budget = 10 * dim + 50  # Santoni et al. 2023's budget rule
        self.name = label or (f"syn_{fname}_d{dim}_k{conditioning:g}_n{noise:g}_{vartype}_a{active:g}")
        self.reseed(seed)

    # ---- structure -------------------------------------------------------------------------
    def reseed(self, seed: int) -> None:
        """Build the instance for this seed. The active and discrete coordinates are
        fixed per task; the seed picks everything else."""
        self.seed = int(seed)
        d = self.dim
        # Fixed per task: depends on the task's settings, not the seed.
        fixed = np.random.default_rng([self.function_id, d, round(100 * self.active_share),
                                       ["continuous", "integer", "categorical"].index(self.vartype), 20261109])
        n_active = max(2, int(round(self.active_share * d)))
        discrete = np.sort(fixed.choice(d, int(round(DISCRETE_SHARE * d)), replace=False)) \
            if self.vartype != "continuous" else np.array([], dtype=int)
        self.active = np.sort(fixed.choice(d, n_active, replace=False))
        continuous = np.setdiff1d(np.arange(d), discrete)
        if len(continuous) and not np.isin(self.active, continuous).any():
            # Keep one continuous coordinate active, so the effective space is never a
            # handful of discrete configurations.
            self.active[fixed.integers(n_active)] = fixed.choice(continuous)
            self.active = np.sort(self.active)
        rng = np.random.default_rng([self.function_id, d, self.seed, 20261109])
        self._problem = ioh.get_problem(self.function_id, instance=self.seed + 1, dimension=n_active,
                                        problem_class=ioh.ProblemClass.BBOB)
        self.f_opt = float(self._problem.optimum.y)
        x_star = np.asarray(self._problem.optimum.x, dtype=float)

        # Discrete coordinates: which ones, how many levels, and (categorical) the label order.
        self.levels = np.zeros(d, dtype=int)  # 0 = continuous
        self.perm = [None] * d
        if self.vartype != "continuous":
            for j, i in enumerate(discrete):
                self.levels[i] = ARITIES[j % len(ARITIES)]
                if self.vartype == "categorical":
                    self.perm[i] = rng.permutation(self.levels[i])
        self.bounds = np.array([[-5.0, 5.0] if k == 0 else [-0.5, k - 0.5] for k in self.levels])

        # Shift each active discrete coordinate so the optimum lies on its nearest grid level.
        self.shift = np.zeros(n_active)
        for a, i in enumerate(self.active):
            k = self.levels[i]
            if k:
                grid = self._grid(k)
                self.shift[a] = x_star[a] - grid[np.argmin(np.abs(grid - x_star[a]))]
        self.x_star = x_star

        # Conditioning: compress toward x* along random directions (curvature ratio kappa).
        if self.kappa > 1:
            q, r = np.linalg.qr(rng.standard_normal((n_active, n_active)))
            q *= np.sign(np.diag(r))
            lam = self.kappa ** (-0.5 * np.arange(n_active) / max(n_active - 1, 1))
            self.transform = q @ np.diag(lam) @ q.T
        else:
            self.transform = None

        self.noise_rng = np.random.default_rng([self.seed, 7])
        self.noise_scale = self.good_point_gap(rng) if self.noisy else 0.0
        self.sigma = self.noise_level * self.noise_scale

    def good_point_gap(self, rng) -> float:
        """The gap f - f_opt of a point better than 90% of uniformly random points."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        gaps = np.array([self._f_true(x) for x in rng.uniform(lo, hi, size=(NOISE_SAMPLES, self.dim))])
        return float(np.quantile(gaps - self.f_opt, GOOD_POINT_QUANTILE))

    @staticmethod
    def _grid(k: int) -> np.ndarray:
        return -5.0 + 10.0 * (np.arange(k) + 0.5) / k

    # ---- evaluation ------------------------------------------------------------------------
    def to_base(self, x) -> np.ndarray:
        """Public point -> the BBOB function's point (active coordinates only)."""
        x = np.asarray(x, dtype=float)
        z = np.empty(len(self.active))
        for a, i in enumerate(self.active):
            k = self.levels[i]
            if k:
                idx = int(np.clip(np.rint(x[i]), 0, k - 1))
                level = self.perm[i][idx] if self.perm[i] is not None else idx
                z[a] = self._grid(k)[level] + self.shift[a]
            else:
                z[a] = x[i]
        if self.transform is not None:
            z = self.x_star + self.transform @ (z - self.x_star)
        return z

    def _f_true(self, x) -> float:
        return float(self._problem(self.to_base(x)))

    def _f(self, x) -> float:
        y = self._f_true(x)
        return y + self.sigma * float(self.noise_rng.standard_normal()) if self.noisy else y

    # ---- public description ---------------------------------------------------------------
    @property
    def variables(self) -> list[dict]:
        """problem.yaml entries for x0.. (the runner adds the names)."""
        out = []
        for i, k in enumerate(self.levels):
            if k == 0:
                out.append({"type": "continuous", "low": -5.0, "high": 5.0})
            elif self.vartype == "integer":
                out.append({"type": "integer", "low": 0, "high": int(k - 1)})
            else:
                out.append({"type": "categorical", "choices": category_labels(k)})
        return out

    def x_star_public(self) -> np.ndarray:
        """A public point attaining f_opt (used to check the construction). The
        conditioning transform fixes x*, so only the discrete mapping is inverted."""
        x = np.zeros(self.dim)
        for a, i in enumerate(self.active):
            k = self.levels[i]
            if k:
                level = int(np.argmin(np.abs(self._grid(k) + self.shift[a] - self.x_star[a])))
                x[i] = int(np.argsort(self.perm[i])[level]) if self.perm[i] is not None else level
            else:
                x[i] = self.x_star[a]
        return x


def category_labels(k: int) -> list[str]:
    return [f"c{j:02d}" for j in range(k)]


def l18_tasks() -> dict:
    """Registry entries for the 18 L18 cells: key -> factory."""
    out = {}
    for n, row in enumerate(L18, 1):
        which, mod, cond, dim, noise, vtype, active = row[:7]
        fid, fname = FUNCTIONS[MODALITY[mod]][which - 1]
        spec = dict(function_id=fid, fname=fname, dim=DIMENSION[LEVELS[dim]],
                    conditioning=CONDITIONING[LEVELS[cond]], noise=NOISE[LEVELS[noise]],
                    vartype=VARTYPE[LEVELS[vtype]], active=ACTIVE[LEVELS[active]])
        key = f"l18_{n:02d}"
        out[key] = (lambda s=spec, k=key: SyntheticTask(**s, label=f"{k}_{s['fname']}"))
    return out
