"""Bayesian optimization with a GP (botorch SingleTaskGP, Matern-5/2 ARD,
noise inferred), refitted on all data at every proposal.

args:
  n_init        Sobol points first, counting points already told (default 2 * (dim + 1)).
  acquisition   "logei" (log expected improvement, default) or "ucb".
  beta          UCB exploration weight (default 2.0).
  num_restarts, raw_samples   acquisition optimizer settings (defaults 10, 512).
On an indexed pool the acquisition is maximized over the unused candidates.
Model state is not saved: each proposal refits from the data, so `continue`
and `restart` with warm_start differ only in the Sobol stream and counters.
Cost grows cubically with the number of points; slow beyond ~1000.
"""
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from scipy.stats import qmc

import _gp
from _base import Base


class Optimizer(Base):
    max_ask = 1  # proposes one point at a time; refits or steps on each result
    def __init__(self, space, seed=0, n_init=None, acquisition="logei", beta=2.0,
                 num_restarts=10, raw_samples=512):
        super().__init__(space, seed)
        if acquisition not in ("logei", "ucb"):
            raise ValueError("acquisition must be logei or ucb")
        self.n_init = n_init if n_init is not None else 2 * (self.dim + 1)
        self.acquisition, self.beta = acquisition, beta
        self.num_restarts, self.raw_samples = num_restarts, raw_samples
        self.sobol, self.calls = qmc.Sobol(self.dim, seed=seed), 0

    def propose(self):
        self.calls += 1
        if len(self.y) < self.n_init:
            return self.sobol.random(1)[0]
        model = _gp.fit(self.U, -_gp.model_values(self.y), self.seed + self.calls)  # botorch maximizes
        if self.acquisition == "logei":
            acq = LogExpectedImprovement(model, best_f=model.train_targets.max())
        else:
            acq = UpperConfidenceBound(model, beta=self.beta)
        if self.indexed:
            free = [i for i in range(len(self.cands)) if i not in self.used and i not in self.pending]
            if not free:
                return None
            with torch.no_grad():
                scores = acq(torch.tensor(self.cands[free]).unsqueeze(1))
            return free[int(scores.argmax())]
        torch.manual_seed(self.seed + self.calls)
        bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
        x, _ = optimize_acqf(acq, bounds, q=1, num_restarts=self.num_restarts, raw_samples=self.raw_samples)
        return x[0].detach().numpy()
