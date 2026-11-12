"""TuRBO-1 style trust-region BO with a noise-aware GP and Thompson sampling.

State is plain data (all X, y, trust-region counters, rng state), so save/load
is a JSON dump; the GP is refit from scratch on every ask.
"""
import json
import math
import os

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, length_init=0.8, length_min=0.5 ** 7,
                 length_max=1.6, succ_tol=3, fail_tol=None, n_cand=4000,
                 n_init=0, **_):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.dim = len(self.lo)
        self.seed = seed
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.succ_tol = succ_tol
        self.fail_tol = fail_tol if fail_tol is not None else max(4, self.dim // 2)
        self.n_cand = n_cand
        self.n_init = n_init
        self.length = length_init
        self.succ = 0
        self.fail = 0
        self.restarts = 0
        self.X = []          # unit-cube points, all observations
        self.y = []
        self.tr_start = 0    # index where the current trust region's data begins
        self.pending = []
        self.rng_state = np.random.default_rng(seed).bit_generator.state

    # ---- helpers
    def _rng(self):
        r = np.random.default_rng()
        r.bit_generator.state = self.rng_state
        return r

    def _to_unit(self, x):
        return (np.asarray(x, float) - self.lo) / (self.hi - self.lo)

    def _from_unit(self, u):
        return self.lo + np.asarray(u) * (self.hi - self.lo)

    def _fit(self, X, y):
        Xt = torch.tensor(X)
        yt = torch.tensor(y).unsqueeze(-1)
        lik = GaussianLikelihood(noise_constraint=Interval(1e-6, 1.0))
        kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.dim,
                                        lengthscale_constraint=Interval(0.005, 4.0)),
                           outputscale_constraint=Interval(0.05, 20.0))
        gp = SingleTaskGP(Xt, yt, likelihood=lik, covar_module=kern,
                          outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        gp.eval()
        return gp

    # ---- interface
    def ask(self, n):
        rng = self._rng()
        out = []
        for _ in range(n):
            Xa = np.array(self.X[self.tr_start:])
            ya = np.array(self.y[self.tr_start:])
            if len(ya) < max(self.n_init, 2):
                u = rng.random(self.dim)
            else:
                try:
                    u = self._propose(Xa, ya, rng)
                except Exception:
                    # model failure: random point near the best observed point
                    c = Xa[np.argmin(ya)]
                    u = np.clip(c + self.length * (rng.random(self.dim) - 0.5), 0, 1)
            self.pending.append(u.tolist())
            out.append(self._from_unit(u).tolist())
        self.rng_state = rng.bit_generator.state
        return out

    def _propose(self, Xa, ya, rng):
        # Clip extreme values in the model's view (helps with heavy tails).
        med = np.median(ya)
        mad = np.median(np.abs(ya - med)) + 1e-9
        yc = np.minimum(ya, med + 5 * mad)
        gp = self._fit(Xa, yc)
        with torch.no_grad():
            mu = gp.posterior(torch.tensor(Xa)).mean.squeeze(-1).numpy()
        center = Xa[np.argmin(mu)]
        ls = gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        w = ls / np.exp(np.mean(np.log(ls)))
        lb = np.clip(center - w * self.length / 2, 0, 1)
        ub = np.clip(center + w * self.length / 2, 0, 1)
        pert = lb + (ub - lb) * rng.random((self.n_cand, self.dim))
        prob = min(20.0 / self.dim, 1.0)
        mask = rng.random((self.n_cand, self.dim)) <= prob
        none = ~mask.any(1)
        mask[none, rng.integers(0, self.dim, none.sum())] = True
        cand = np.where(mask, pert, center)
        with torch.no_grad():
            torch.manual_seed(int(rng.integers(0, 2 ** 31)))
            post = gp.posterior(torch.tensor(cand))
            s = post.rsample(torch.Size([1])).squeeze(0).squeeze(-1).numpy()
        return cand[np.argmin(s)]

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            u = self._to_unit(x)
            v = float(v)
            if not math.isfinite(v):
                v = max(self.y) if self.y else 1e6
            prev_best = min(self.y[self.tr_start:]) if len(self.y) > self.tr_start else math.inf
            self.X.append(u.tolist())
            self.y.append(v)
            # match against pending asks; only own asks drive the trust region
            hit = None
            for i, p in enumerate(self.pending):
                if np.allclose(p, u, atol=1e-9):
                    hit = i
                    break
            if hit is None:
                continue
            self.pending.pop(hit)
            if not math.isfinite(prev_best):
                continue
            if v < prev_best - 1e-3 * abs(prev_best):
                self.succ += 1
                self.fail = 0
            else:
                self.succ = 0
                self.fail += 1
            if self.succ >= self.succ_tol:
                self.length = min(2 * self.length, self.length_max)
                self.succ = 0
            elif self.fail >= self.fail_tol:
                self.length /= 2
                self.fail = 0
            if self.length < self.length_min:
                # restart: fresh trust region with a few random points
                self.restarts += 1
                self.length = self.length_init
                self.succ = self.fail = 0
                self.tr_start = len(self.X)
                self.n_init = max(self.n_init, self.dim)

    def save(self, path):
        d = dict(self.__dict__)
        d["lo"] = self.lo.tolist()
        d["hi"] = self.hi.tolist()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls.__new__(cls)
        o.__dict__.update(d)
        o.lo = np.array(d["lo"])
        o.hi = np.array(d["hi"])
        return o
