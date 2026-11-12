import json
import warnings
warnings.filterwarnings("ignore")
import math

import numpy as np
import torch
from scipy.stats import norm, qmc


class Optimizer:
    """TuRBO-1 style trust-region BO for a noisy, continuous box.

    y is rank (Gaussian copula) transformed before the GP fit, which copes with a
    huge dynamic range. The trust region is centred on the best GP posterior mean
    among observed points (robust to noise); candidates come from Thompson sampling
    on Sobol perturbations inside the region. Success/failure counters resize it;
    on collapse it restarts around the incumbent with the initial length.
    """

    def __init__(self, space, seed=0, length_init=0.4, length_min=0.01,
                 length_max=1.0, n_cand=2048, max_fit=400, fail_tol=None,
                 succ_tol=3):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.d = len(self.bounds)
        self.seed = seed
        self.length_init = length_init
        self.length = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.n_cand = n_cand
        self.max_fit = max_fit
        self.fail_tol = fail_tol or max(4, self.d)
        self.succ_tol = succ_tol
        self.succ = 0
        self.fail = 0
        self.X = []
        self.y = []
        self.best_seen = None
        self.n_restarts = 0
        self.rng = np.random.default_rng(seed)

    # unit cube helpers
    def _to_u(self, X):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(X) - lo) / (hi - lo)

    def _from_u(self, U):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.asarray(U) * (hi - lo)

    def _fit(self, U, z):
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.constraints import Interval
        from gpytorch.likelihoods import GaussianLikelihood
        Xt = torch.tensor(U, dtype=torch.double)
        Yt = torch.tensor(z, dtype=torch.double).unsqueeze(-1)
        lik = GaussianLikelihood(noise_constraint=Interval(1e-6, 1.0))
        kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.d,
                                        lengthscale_constraint=Interval(0.005, 4.0)))
        gp = SingleTaskGP(Xt, Yt, likelihood=lik, covar_module=kern)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        return gp

    def ask(self, n):
        U = self._to_u(np.array(self.X)) if self.X else np.zeros((0, self.d))
        y = np.array(self.y, dtype=float)
        out = []
        if len(y) < 2 * self.d:
            s = qmc.Sobol(self.d, scramble=True, seed=int(self.rng.integers(1 << 31)))
            return [list(map(float, p)) for p in self._from_u(s.random(n))]
        # copula transform
        r = (np.argsort(np.argsort(y)) + 0.5) / len(y)
        z = norm.ppf(r)
        # centre: best posterior mean among observed points
        ib = int(np.argmin(y))
        # local subset for fitting
        if len(y) > self.max_fit:
            dist = np.linalg.norm(U - U[ib], axis=1)
            idx = np.argsort(dist)[:self.max_fit]
        else:
            idx = np.arange(len(y))
        gp = self._fit(U[idx], z[idx])
        with torch.no_grad():
            mu = gp.posterior(torch.tensor(U[idx])).mean.squeeze(-1).numpy()
        c = U[idx][int(np.argmin(mu))]
        ls = gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        w = ls / np.exp(np.mean(np.log(ls)))
        half = np.clip(w * self.length / 2, 0, 0.5)
        lo = np.clip(c - half, 0, 1)
        hi = np.clip(c + half, 0, 1)
        s = qmc.Sobol(self.d, scramble=True, seed=int(self.rng.integers(1 << 31)))
        pert = lo + (hi - lo) * s.random(self.n_cand)
        prob = min(20.0 / self.d, 1.0)
        mask = self.rng.random((self.n_cand, self.d)) <= prob
        none = ~mask.any(1)
        mask[none, self.rng.integers(0, self.d, none.sum())] = True
        cand = np.where(mask, pert, c)
        Xc = torch.tensor(cand, dtype=torch.double)
        with torch.no_grad():
            post = gp.posterior(Xc)
            chosen = []
            for _ in range(n):
                f = post.rsample(torch.Size([1])).squeeze(0).squeeze(-1).numpy()
                f[chosen] = np.inf
                chosen.append(int(np.argmin(f)))
        out = self._from_u(cand[chosen])
        return [list(map(float, p)) for p in out]

    def tell(self, X, y, g):
        y = [float(v) for v in y]
        if self.best_seen is not None and len(y) <= 4:
            # update trust region (normal one-at-a-time operation)
            if min(y) < self.best_seen - 1e-3 * abs(self.best_seen):
                self.succ += 1
                self.fail = 0
            else:
                self.fail += 1
                self.succ = 0
            if self.succ >= self.succ_tol:
                self.length = min(2 * self.length, self.length_max)
                self.succ = 0
            elif self.fail >= self.fail_tol:
                self.length /= 2
                self.fail = 0
            if self.length < self.length_min:
                self.length = self.length_init
                self.n_restarts += 1
        self.X += [list(map(float, x)) for x in X]
        self.y += y
        self.best_seen = min(self.y)

    def save(self, path):
        d = {k: getattr(self, k) for k in
             ["seed", "length_init", "length", "length_min", "length_max", "n_cand",
              "max_fit", "fail_tol", "succ_tol", "succ", "fail", "X", "y",
              "best_seen", "n_restarts"]}
        d["bounds"] = self.bounds.tolist()
        d["rng"] = self.rng.bit_generator.state
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls({"bounds": d["bounds"]})
        for k, v in d.items():
            if k not in ("bounds", "rng"):
                setattr(o, k, v)
        o.rng = np.random.default_rng()
        o.rng.bit_generator.state = d["rng"]
        return o
