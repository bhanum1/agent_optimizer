"""TuRBO-1 style trust-region BO, sequential, with a log transform of y.

GP (ARD Matern-5/2, fitted noise) on z = log(y - min(y) + delta), standardized,
using the current run's points (nearest `max_fit` to the TR center).
Thompson sampling on perturbation-masked candidates inside the trust region.
TR is centered at the run's observed point with the best posterior mean (noise-robust).
On collapse (L < L_min), a new run starts with `n_init` Sobol points; global data kept
for the record but each run fits its own data.
"""
import math
import pickle
import warnings

import numpy as np
from scipy.stats import qmc

warnings.filterwarnings("ignore")


class Optimizer:
    def __init__(self, space, seed=0, n_init=10, L_init=0.8, L_min=0.0078125,
                 L_max=1.6, succ_tol=3, fail_tol=5, n_cand=3000, max_fit=400):
        self.lb = np.array([b[0] for b in space["bounds"]], float)
        self.ub = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lb)
        self.rng = np.random.default_rng(seed)
        self.n_init = int(n_init)
        self.L_init, self.L_min, self.L_max = float(L_init), float(L_min), float(L_max)
        self.succ_tol, self.fail_tol = int(succ_tol), int(fail_tol)
        self.n_cand, self.max_fit = int(n_cand), int(max_fit)
        self.X = np.zeros((0, self.d))   # unit cube, all data
        self.y = np.zeros(0)
        self.run = np.zeros(0, int)      # run id of each point
        self.run_id = 0
        self.L = self.L_init
        self.succ = 0
        self.fail = 0
        self.init_queue = []             # unit points still to evaluate for current run init
        self.pending = []
        self.n_restarts = 0

    # ---------- helpers
    def _to_unit(self, X):
        return (np.asarray(X, float) - self.lb) / (self.ub - self.lb)

    def _from_unit(self, U):
        return self.lb + np.asarray(U) * (self.ub - self.lb)

    def _transform(self, y):
        ymin = y.min()
        spread = np.median(y) - ymin
        delta = max(0.05 * spread, 1e-12, 1e-9 * abs(ymin))
        z = np.log(y - ymin + delta)
        return (z - z.mean()) / (z.std() + 1e-12)

    def _new_run_init(self):
        s = qmc.Sobol(d=self.d, scramble=True, seed=int(self.rng.integers(2**31)))
        self.init_queue = s.random(self.n_init).tolist()

    def _run_mask(self):
        m = self.run == self.run_id
        return m

    def _fit(self, Xr, zr, center):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.constraints import Interval
        from gpytorch.likelihoods import GaussianLikelihood
        torch.manual_seed(int(self.rng.integers(2**31)))
        if center is not None and len(Xr) > self.max_fit:
            idx = np.argsort(np.linalg.norm(Xr - center, axis=1))[:self.max_fit]
            Xr, zr = Xr[idx], zr[idx]
        Xt = torch.tensor(Xr, dtype=torch.double)
        yt = torch.tensor(zr, dtype=torch.double).unsqueeze(-1)
        lik = GaussianLikelihood(noise_constraint=Interval(1e-6, 1.0))
        cov = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.d,
                                       lengthscale_constraint=Interval(0.005, 4.0)),
                          outputscale_constraint=Interval(0.05, 20.0))
        model = SingleTaskGP(Xt, yt, likelihood=lik, covar_module=cov)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        return model

    # ---------- interface
    def ask(self, n):
        out = []
        for _ in range(n):
            u = self._ask_one()
            self.pending.append(u)
            out.append(self._from_unit(u).tolist())
        return out

    def _ask_one(self):
        import torch
        if not self.init_queue and self._run_mask().sum() == 0 and not self.pending:
            self._new_run_init()
        if self.init_queue:
            return np.array(self.init_queue.pop(0))
        m = self._run_mask()
        Xr, yr = self.X[m], self.y[m]
        if len(Xr) < 2:
            return self.rng.random(self.d)
        zr = self._transform(yr)
        model = self._fit(Xr, zr, None)
        with torch.no_grad():
            mu = model.posterior(torch.tensor(Xr)).mean.squeeze(-1).numpy()
        center = Xr[int(np.argmin(mu))]
        if len(Xr) > self.max_fit:
            model = self._fit(Xr, zr, center)
        ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        w = ls / ls.mean()
        w = w / np.prod(w) ** (1.0 / self.d)
        half = w * self.L / 2.0
        tlb = np.clip(center - half, 0, 1)
        tub = np.clip(center + half, 0, 1)
        s = qmc.Sobol(d=self.d, scramble=True, seed=int(self.rng.integers(2**31)))
        pert = tlb + (tub - tlb) * s.random(self.n_cand)
        prob = min(20.0 / self.d, 1.0)
        mask = self.rng.random((self.n_cand, self.d)) <= prob
        none = ~mask.any(axis=1)
        mask[none, self.rng.integers(0, self.d, none.sum())] = True
        cand = np.where(mask, pert, center)
        with torch.no_grad():
            post = model.posterior(torch.tensor(cand))
            samp = post.rsample(torch.Size([1])).squeeze(0).squeeze(-1).numpy()
        return cand[int(np.argmin(samp))]

    def tell(self, X, y, g):
        U = self._to_unit(X)
        y = np.asarray(y, float)
        for u, v in zip(U, y):
            # match pending
            k = None
            for i, p in enumerate(self.pending):
                if np.allclose(p, u, atol=1e-9):
                    k = i
                    break
            ours = k is not None
            if ours:
                self.pending.pop(k)
            m = self._run_mask()
            prev = self.y[m]
            self.X = np.vstack([self.X, u])
            self.y = np.append(self.y, v)
            self.run = np.append(self.run, self.run_id)
            if not ours:
                continue  # warm-start / foreign data: just record
            if len(prev) < self.n_init or (self.init_queue):
                continue
            allr = self.y[self._run_mask()]
            z = self._transform(allr)
            zbest_prev = z[:-1].min()
            if z[-1] < zbest_prev - 1e-3:
                self.succ += 1
                self.fail = 0
            else:
                self.fail += 1
                self.succ = 0
            if self.succ >= self.succ_tol:
                self.L = min(2.0 * self.L, self.L_max)
                self.succ = 0
            elif self.fail >= self.fail_tol:
                self.L /= 2.0
                self.fail = 0
            if self.L < self.L_min:
                self.run_id += 1
                self.n_restarts += 1
                self.L = self.L_init
                self.succ = self.fail = 0
                self._new_run_init()

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
