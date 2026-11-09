"""Bayesian optimization for mixed spaces (botorch MixedSingleTaskGP), refitted at every proposal.

Models each variable in its own terms rather than through the encoding:
continuous on its (log) scale, integer and ordinal as ordered discrete values,
categorical with a categorical kernel. The acquisition (log expected
improvement) is optimized by alternating discrete local search and continuous
L-BFGS (botorch optimize_acqf_mixed_alternating). With only continuous
variables it is the same model as gp_bo.

args:
  n_init        Sobol points first, counting points already told (default 2 * (variables + 1)).
  num_restarts, raw_samples   acquisition optimizer settings (defaults 10, 512).
Box spaces only (use gp_bo on a candidate pool). Model state is not saved: each
proposal refits from the data. Cost grows cubically with the number of points.
"""
import warnings

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

import _gp
from _base import Base

torch.set_default_dtype(torch.float64)


class Optimizer(Base):
    max_ask = 1

    def __init__(self, space, seed=0, n_init=None, num_restarts=10, raw_samples=512):
        super().__init__(space, seed)
        if self.indexed:
            raise ValueError("mixed_bo works on a box; use gp_bo for a candidate pool")
        self.vars = space.get("variables") or [
            {"name": f"x{i}", "type": "continuous", "coords": [i], "low": lo, "high": hi, "log": False}
            for i, (lo, hi) in enumerate(space["bounds"])]
        self.k = len(self.vars)
        self.n_init = n_init if n_init is not None else 2 * (self.k + 1)
        self.num_restarts, self.raw_samples = num_restarts, raw_samples
        self.sobol, self.calls = qmc.Sobol(self.k, seed=seed), 0
        self.discrete, self.categorical = {}, {}
        for j, v in enumerate(self.vars):
            if v["type"] == "categorical":
                self.categorical[j] = [float(i) for i in range(len(v["choices"]))]
            elif v["type"] == "ordinal":
                n = len(v["choices"])
                self.discrete[j] = [i / (n - 1) for i in range(n)]
            elif v["type"] == "integer":
                values = np.arange(int(v["low"]), int(v["high"]) + 1, dtype=float)
                if len(values) <= 200:
                    self.discrete[j] = [float(self._scale(v, x)) for x in values]
        self.upper = [len(self.vars[j]["choices"]) - 1.0 if j in self.categorical else 1.0 for j in range(self.k)]

    # ---- per-variable coordinates z: [0, 1] scale, or a category index ---------------------------
    @staticmethod
    def _scale(v, value):
        if v.get("log"):
            return (np.log10(value) - np.log10(v["low"])) / (np.log10(v["high"]) - np.log10(v["low"]))
        return (value - v["low"]) / (v["high"] - v["low"])

    def to_z(self, x):
        z = np.empty(self.k)
        for j, v in enumerate(self.vars):
            c = v["coords"]
            if v["type"] == "categorical":
                z[j] = float(np.argmax(x[c]))
            elif v["type"] == "ordinal":
                z[j] = np.clip(round(x[c[0]]), 0, len(v["choices"]) - 1) / (len(v["choices"]) - 1)
            else:
                value = 10 ** x[c[0]] if v.get("log") else x[c[0]]
                if v["type"] == "integer":
                    value = np.clip(round(value), v["low"], v["high"])
                z[j] = np.clip(self._scale(v, value), 0, 1)
        return z

    def to_x(self, z):
        x = np.zeros(self.dim)
        for j, v in enumerate(self.vars):
            c = v["coords"]
            if v["type"] == "categorical":
                x[c[int(round(z[j]))]] = 1.0
            elif v["type"] == "ordinal":
                x[c[0]] = round(z[j] * (len(v["choices"]) - 1))
            else:
                if v.get("log"):
                    value = 10 ** (np.log10(v["low"]) + z[j] * (np.log10(v["high"]) - np.log10(v["low"])))
                else:
                    value = v["low"] + z[j] * (v["high"] - v["low"])
                if v["type"] == "integer":
                    value = float(np.clip(round(value), v["low"], v["high"]))
                x[c[0]] = np.log10(value) if v.get("log") else value
        return x

    def _snap(self, u):
        z = np.array(u, dtype=float)
        for j, values in self.categorical.items():
            z[j] = min(int(u[j] * len(values)), len(values) - 1)
        for j, values in self.discrete.items():
            z[j] = values[int(np.argmin(np.abs(np.array(values) - u[j])))]
        return z

    # ---- protocol -------------------------------------------------------------------------------
    def propose(self):
        self.calls += 1
        if len(self.y) < self.n_init:
            z = self._snap(self.sobol.random(1)[0])
        else:
            z = self._acquire()
        return self.to_unit(self.to_x(z))

    def _acquire(self):
        Z = torch.tensor(np.array([self.to_z(self.from_unit(u)) for u in self.U]))
        Y = torch.tensor(-_gp.model_values(self.y)).unsqueeze(-1)
        torch.manual_seed(self.seed + self.calls)
        if self.categorical:
            model = MixedSingleTaskGP(Z, Y, cat_dims=sorted(self.categorical), outcome_transform=Standardize(1))
        else:
            model = SingleTaskGP(Z, Y, outcome_transform=Standardize(1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            acq = LogExpectedImprovement(model, best_f=Y.max())
            bounds = torch.tensor([[0.0] * self.k, self.upper])
            if self.categorical or self.discrete:
                z, _ = optimize_acqf_mixed_alternating(acq, bounds, discrete_dims=self.discrete or None,
                                                       cat_dims=self.categorical or None,
                                                       num_restarts=self.num_restarts, raw_samples=self.raw_samples)
            else:
                z, _ = optimize_acqf(acq, bounds, q=1, num_restarts=self.num_restarts, raw_samples=self.raw_samples)
        z = z[0].detach().numpy().copy()
        for j, values in self.categorical.items():
            z[j] = float(np.clip(round(z[j]), 0, len(values) - 1))
        for j, values in self.discrete.items():
            z[j] = values[int(np.argmin(np.abs(np.array(values) - z[j])))]
        return z
