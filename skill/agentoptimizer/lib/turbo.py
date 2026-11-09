"""TuRBO-1 (Eriksson et al., 2019): one trust region around the best point,
Thompson sampling from a local GP, sequential (batch size 1).

args:
  n_init        Sobol points at the start of each trust region, counting
                points already told on the first one (default 2 * dim).
  length_init, length_min, length_max   trust-region side in the unit cube
                (defaults 0.8, 0.5**7, 1.6).
  success_tol   consecutive improvements before doubling (default 3).
  failure_tol   consecutive failures before halving (default max(4, dim)).
  n_candidates  Thompson-sampling candidates per proposal (default min(100 * dim, 2000)).
When the region shrinks below length_min the method restarts with a new
region and a fresh design; the GP uses only the current region's points.
On an indexed pool, unused candidates inside the region are sampled.
"""
import numpy as np
import torch
from botorch.generation import MaxPosteriorSampling
from scipy.stats import qmc

import _gp
from _base import Base


class Optimizer(Base):
    def __init__(self, space, seed=0, n_init=None, length_init=0.8, length_min=0.5 ** 7,
                 length_max=1.6, success_tol=3, failure_tol=None, n_candidates=None):
        super().__init__(space, seed)
        self.n_init = n_init if n_init is not None else 2 * self.dim
        self.length_init, self.length_min, self.length_max = length_init, length_min, length_max
        self.success_tol = success_tol
        self.failure_tol = failure_tol if failure_tol is not None else max(4, self.dim)
        self.n_candidates = n_candidates or min(100 * self.dim, 2000)
        self.sobol, self.calls, self.restarts = qmc.Sobol(self.dim, seed=seed), 0, 0
        self.new_region(keep_told=True)

    def new_region(self, keep_told=False):
        self.local = list(range(len(self.y))) if keep_told else []
        self.length, self.successes, self.failures = self.length_init, 0, 0
        self.region_best = min(self.y[self.local]) if self.local else np.inf

    def absorb(self, u, value):
        self.local.append(len(self.y) - 1)
        self.region_best = min(self.region_best, value)

    def propose(self):
        self.calls += 1
        if len(self.local) < self.n_init:
            return self.sobol.random(1)[0]
        U, y = self.U[self.local], self.y[self.local]
        center = U[int(np.argmin(y))]
        model = _gp.fit(U, -_gp.model_values(y), self.seed + self.calls)
        scales = model.covar_module.lengthscale.detach().numpy().ravel()
        scales = scales / np.prod(scales) ** (1 / self.dim)
        lo = np.clip(center - scales * self.length / 2, 0, 1)
        hi = np.clip(center + scales * self.length / 2, 0, 1)
        if self.indexed:
            free = np.array([i for i in range(len(self.cands)) if i not in self.used and i not in self.pending])
            if not len(free):
                return None
            inside = free[((self.cands[free] >= lo) & (self.cands[free] <= hi)).all(1)]
            index = inside if len(inside) else free
            if len(index) > self.n_candidates:
                index = self.rng.choice(index, self.n_candidates, replace=False)
            torch.manual_seed(self.seed + self.calls)
            with torch.no_grad():
                x = MaxPosteriorSampling(model, replacement=False)(torch.tensor(self.cands[index]), num_samples=1)
            hit = np.flatnonzero((self.cands[index] == x[0].numpy()).all(1))
            return int(index[hit[0]])
        else:
            pert = lo + (hi - lo) * qmc.Sobol(self.dim, seed=self.seed + self.calls).random(self.n_candidates)
            prob = min(20.0 / self.dim, 1.0)
            mask = self.rng.random((self.n_candidates, self.dim)) <= prob
            mask[np.arange(self.n_candidates), self.rng.integers(self.dim, size=self.n_candidates)] = True
            pool = np.where(mask, pert, center)
        torch.manual_seed(self.seed + self.calls)
        with torch.no_grad():
            x = MaxPosteriorSampling(model, replacement=False)(torch.tensor(pool), num_samples=1)
        return x[0].numpy()

    def observe(self, u, value):
        self.local.append(len(self.y) - 1)
        if len(self.local) <= self.n_init:
            self.region_best = min(self.region_best, value)
            return
        if value < self.region_best - 1e-3 * abs(self.region_best):
            self.successes, self.failures = self.successes + 1, 0
        else:
            self.successes, self.failures = 0, self.failures + 1
        self.region_best = min(self.region_best, value)
        if self.successes == self.success_tol:
            self.length, self.successes = min(2 * self.length, self.length_max), 0
        elif self.failures == self.failure_tol:
            self.length, self.failures = self.length / 2, 0
        if self.length < self.length_min:
            self.restarts += 1
            self.new_region()
