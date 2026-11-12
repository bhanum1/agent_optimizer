import json
import math
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)


class Optimizer:
    """GP-BO with qLogNEI, optionally restricted to a TuRBO-style trust region
    (in the unit cube) centred on the observed point with best posterior mean."""

    def __init__(self, space, seed=0, tr=True, L0=0.8, Lmin=0.01, Lmax=1.6,
                 succ_tol=3, fail_tol=10, n_init=10):
        self.lb = np.array([b[0] for b in space["bounds"]], float)
        self.ub = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lb)
        self.seed = seed
        self.tr = tr
        self.L0, self.L, self.Lmin, self.Lmax = L0, L0, Lmin, Lmax
        self.succ_tol, self.fail_tol = succ_tol, fail_tol
        self.n_succ = self.n_fail = 0
        self.n_init = n_init
        self.X, self.y = [], []
        self.n_asked = 0
        self.best_y = None

    def _to_unit(self, X):
        return (np.asarray(X, float) - self.lb) / (self.ub - self.lb)

    def _from_unit(self, U):
        return self.lb + np.asarray(U, float) * (self.ub - self.lb)

    def ask(self, n):
        out = []
        for _ in range(n):
            out.append(self._ask_one())
            self.n_asked += 1
        return out

    def _ask_one(self):
        rng = np.random.default_rng(self.seed * 100003 + self.n_asked)
        if len(self.y) < self.n_init:
            return list(self._from_unit(rng.random(self.d)))
        torch.manual_seed(self.seed * 100003 + self.n_asked)
        U = torch.tensor(self._to_unit(self.X))
        Y = torch.tensor(np.asarray(self.y, float)).unsqueeze(-1)
        model = SingleTaskGP(U, Y, input_transform=Normalize(self.d, bounds=torch.stack([torch.zeros(self.d), torch.ones(self.d)])),
                             outcome_transform=Standardize(1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        bounds = torch.stack([torch.zeros(self.d), torch.ones(self.d)])
        if self.tr:
            with torch.no_grad():
                mu = model.posterior(U).mean.squeeze(-1)
            c = U[int(torch.argmin(mu))]
            try:
                ls = model.covar_module.lengthscale.detach().view(-1)
            except AttributeError:
                ls = model.covar_module.base_kernel.lengthscale.detach().view(-1)
            w = ls / ls.mean()
            w = w / torch.prod(w) ** (1.0 / self.d)
            half = w * self.L / 2.0
            bounds = torch.stack([torch.clamp(c - half, 0, 1), torch.clamp(c + half, 0, 1)])
        acq = qLogNoisyExpectedImprovement(model, X_baseline=U, prune_baseline=True)
        try:
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=8, raw_samples=512)
            u = cand.detach().numpy()[0]
        except Exception:
            lo, hi = bounds[0].numpy(), bounds[1].numpy()
            u = lo + rng.random(self.d) * (hi - lo)
        return list(np.clip(self._from_unit(u), self.lb, self.ub))

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            yi = float(yi)
            # bulk tells (warm start) only seed the data, not the TR counters
            if self.best_y is not None and self.tr and len(y) == 1:
                if yi < self.best_y - 1e-3 * abs(self.best_y):
                    self.n_succ += 1
                    self.n_fail = 0
                else:
                    self.n_fail += 1
                    self.n_succ = 0
                if self.n_succ >= self.succ_tol:
                    self.L = min(2 * self.L, self.Lmax)
                    self.n_succ = 0
                elif self.n_fail >= self.fail_tol:
                    self.L = self.L / 2
                    self.n_fail = 0
                if self.L < self.Lmin:
                    self.L = self.L0
            self.best_y = yi if self.best_y is None else min(self.best_y, yi)
            self.X.append([float(v) for v in xi])
            self.y.append(yi)

    def save(self, path):
        st = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        with open(path, "w") as f:
            json.dump(st, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            st = json.load(f)
        o = cls.__new__(cls)
        o.__dict__.update(st)
        o.lb = np.array(st["lb"], float)
        o.ub = np.array(st["ub"], float)
        return o
