"""Exploratory GP-BO for a rugged objective: LogEI on a Matern-5/2 ARD GP with
noise fixed near zero and lengthscales capped at ls_max (unit cube), so wells are
not smoothed away. Candidates closer than min_dist (unit cube) to a measured
point are rejected, so BO cannot spend points polishing a known well.
Warm-start friendly (n_init=0). Deterministic objective, minimized.
State is the told data, outstanding asks and RNG counter, saved as JSON."""
import json
import math

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_init=0, ls_max=0.15, noise=1e-4, min_dist=0.03, **_):
        self.bounds = [list(map(float, b)) for b in space["bounds"]]
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.ls_max = float(ls_max)
        self.noise = float(noise)
        self.min_dist = float(min_dist)
        self.X, self.y = [], []
        self.pending = []
        self.n_asked = 0

    # unit cube <-> bounds
    def _to_unit(self, X):
        lo, hi = np.array(self.bounds).T
        return (np.asarray(X, float) - lo) / (hi - lo)

    def _from_unit(self, U):
        lo, hi = np.array(self.bounds).T
        return lo + np.asarray(U, float) * (hi - lo)

    def _init_point(self, k):
        s = qmc.Sobol(d=len(self.bounds), scramble=True, seed=self.seed)
        pts = s.random(2 ** max(1, math.ceil(math.log2(max(self.n_init, k + 1)))))
        return self._from_unit(pts[k]).tolist()

    def _bo_point(self):
        X = torch.tensor(self._to_unit(self.X + self.pending))
        y = np.array(self.y, float)
        y = np.where(np.isfinite(y), y, np.nanmax(y[np.isfinite(y)]) if np.isfinite(y).any() else 0.0)
        Y = torch.tensor(-y).unsqueeze(-1)  # botorch maximizes
        X = X[: len(self.y)]
        torch.manual_seed(self.seed * 1000 + self.n_asked)
        lik = GaussianLikelihood()
        lik.noise = self.noise
        lik.raw_noise.requires_grad_(False)
        gp = SingleTaskGP(X, Y, likelihood=lik, outcome_transform=Standardize(m=1))
        k = gp.covar_module.base_kernel if hasattr(gp.covar_module, "base_kernel") else gp.covar_module
        k.register_constraint("raw_lengthscale", Interval(0.005, self.ls_max))
        k.lengthscale = 0.5 * self.ls_max
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        acq = LogExpectedImprovement(gp, best_f=Y.max())
        d = X.shape[1]
        b = torch.stack([torch.zeros(d), torch.ones(d)])
        # dense random candidates plus the optimizer's pick; best acq outside min_dist
        rng = np.random.default_rng(self.seed * 1000 + self.n_asked)
        C = rng.random((20000, d))
        try:
            cand, _ = optimize_acqf(acq, bounds=b, q=1, num_restarts=10, raw_samples=512)
            C = np.vstack([cand.detach().numpy(), C])
        except Exception:
            pass
        Xn = X.numpy()
        dist = np.min(np.linalg.norm(C[:, None, :] - Xn[None], axis=2), axis=1)
        C = C[dist >= self.min_dist] if (dist >= self.min_dist).any() else C
        with torch.no_grad():
            a = acq(torch.tensor(C).unsqueeze(1)).numpy()
        u = C[int(np.argmax(a))]
        return self._from_unit(np.clip(u, 0, 1)).tolist()

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.y) + len(self.pending) < self.n_init:
                x = self._init_point(len(self.y) + len(self.pending))
            else:
                x = self._bo_point()
            self.pending.append(x)
            self.n_asked += 1
            out.append(x)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = [float(a) for a in x]
            for i, p in enumerate(self.pending):
                if np.allclose(p, x):
                    self.pending.pop(i)
                    break
            self.X.append(x)
            self.y.append(float(v) if v is not None and math.isfinite(float(v)) else float("nan"))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path) as f:
            o.__dict__.update(json.load(f))
        return o
