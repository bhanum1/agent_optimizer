"""TuRBO-1 style trust-region BO (GP + Thompson sampling), with restarts.

State is plain JSON: all observations, trust-region state, numpy RNG state.
"""
import json
import math

import numpy as np
import torch
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood


class _GP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik, dim):
        super().__init__(x, y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)),
            outputscale_constraint=Interval(0.05, 20.0))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, length_init=0.8, length_min=0.5 ** 7, length_max=1.6,
                 succ_tol=3, fail_tol=None, n_cand=4096, max_train=300, log_y=True,
                 n_init_restart=10):
        self.lb = [float(b[0]) for b in space["bounds"]]
        self.ub = [float(b[1]) for b in space["bounds"]]
        self.dim = len(self.lb)
        self.seed = seed
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.succ_tol = succ_tol
        self.fail_tol = fail_tol if fail_tol is not None else max(4, self.dim)
        self.n_cand = n_cand
        self.max_train = max_train
        self.log_y = log_y
        self.n_init_restart = n_init_restart
        self.X = []          # unit-cube points (all)
        self.y = []
        self.tr_start = 0    # index into X/y where current TR's data begins (after restart)
        self.length = length_init
        self.succ = 0
        self.fail = 0
        self.best_tr = None  # best value within current TR
        self.pending_init = []  # queued init points after a restart
        self.rng = np.random.default_rng(seed)
        self.n_restarts = 0

    # --- helpers ---
    def _to_unit(self, x):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return ((np.asarray(x, float) - lb) / (ub - lb)).tolist()

    def _from_unit(self, u):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return (lb + np.clip(np.asarray(u, float), 0, 1) * (ub - lb)).tolist()

    def _restart(self):
        self.n_restarts += 1
        self.length = self.length_init
        self.succ = self.fail = 0
        self.best_tr = None
        self.tr_start = len(self.X)
        # Sobol-ish random init for new TR
        from scipy.stats import qmc
        s = qmc.Sobol(self.dim, scramble=True, seed=int(self.rng.integers(1 << 31)))
        self.pending_init = s.random(self.n_init_restart).tolist()

    def _fit_and_propose(self):
        X = np.array(self.X)
        y = np.array(self.y)
        # training data: use all data (global info) but cap size to nearest-to-center
        if self.tr_start > 0:
            idx_tr = np.arange(self.tr_start, len(y))
        else:
            idx_tr = np.arange(len(y))
        center = X[idx_tr[np.argmin(y[idx_tr])]]
        if len(y) > self.max_train:
            d = np.linalg.norm(X - center, axis=1)
            sel = np.argsort(d)[: self.max_train]
        else:
            sel = np.arange(len(y))
        Xt, yt = X[sel], y[sel]
        if self.log_y:
            yt = np.log(yt - yt.min() + 1e-3 * (np.std(yt) + 1e-12) + 1e-12)
        mu, sd = yt.mean(), yt.std() + 1e-9
        ytn = (yt - mu) / sd
        torch.manual_seed(int(self.rng.integers(1 << 31)))
        xt = torch.tensor(Xt, dtype=torch.double)
        yy = torch.tensor(ytn, dtype=torch.double)
        lik = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-2)).double()
        model = _GP(xt, yy, lik, self.dim).double()
        model.covar_module.base_kernel.lengthscale = 0.5
        model.covar_module.outputscale = 1.0
        lik.noise = 1e-4
        model.train(); lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        with gpytorch.settings.cholesky_jitter(1e-6):
            for _ in range(60):
                opt.zero_grad()
                loss = -mll(model(xt), yy)
                loss.backward()
                opt.step()
        ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        w = ls / ls.mean()
        w = w / np.prod(w ** (1.0 / self.dim))
        lo = np.clip(center - w * self.length / 2, 0, 1)
        hi = np.clip(center + w * self.length / 2, 0, 1)
        from scipy.stats import qmc
        s = qmc.Sobol(self.dim, scramble=True, seed=int(self.rng.integers(1 << 31)))
        pert = lo + (hi - lo) * s.random(self.n_cand)
        prob = min(20.0 / self.dim, 1.0)
        mask = self.rng.random((self.n_cand, self.dim)) <= prob
        none = np.where(mask.sum(1) == 0)[0]
        mask[none, self.rng.integers(0, self.dim, size=len(none))] = True
        cand = np.where(mask, pert, center)
        model.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-6):
            post = model(torch.tensor(cand, dtype=torch.double))
            try:
                f = post.rsample(torch.Size([1])).numpy().ravel()
            except Exception:
                f = (post.mean - 0.0 * post.variance).numpy().ravel()
        return cand[int(np.argmin(f))].tolist()

    # --- protocol ---
    def ask(self, n):
        out = []
        for _ in range(n):
            if self.pending_init:
                u = self.pending_init.pop(0)
            elif len(self.y) < 2:
                u = self.rng.random(self.dim).tolist()
            else:
                u = self._fit_and_propose()
            out.append(self._from_unit(u))
        return out

    def tell(self, X, y, g):
        if len(X) > 1:  # warm start / bulk data: record, don't adapt the TR
            for x, v in zip(X, y):
                v = float(v) if math.isfinite(float(v)) else 1e10
                self.X.append(self._to_unit(x))
                self.y.append(v)
                self.best_tr = v if self.best_tr is None else min(self.best_tr, v)
            return
        for x, v in zip(X, y):
            v = float(v)
            if not math.isfinite(v):
                v = max(self.y) if self.y else 1e10
            self.X.append(self._to_unit(x))
            self.y.append(v)
            if self.best_tr is None:
                self.best_tr = v
                continue
            if v < self.best_tr - 1e-3 * abs(self.best_tr):
                self.succ += 1
                self.fail = 0
            else:
                self.fail += 1
                self.succ = 0
            self.best_tr = min(self.best_tr, v)
            if self.succ >= self.succ_tol:
                self.length = min(2.0 * self.length, self.length_max)
                self.succ = 0
            elif self.fail >= self.fail_tol:
                self.length /= 2.0
                self.fail = 0
        if self.length < self.length_min and not self.pending_init:
            self._restart()

    def save(self, path):
        d = {k: v for k, v in self.__dict__.items() if k != "rng"}
        d["rng"] = self.rng.bit_generator.state
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls.__new__(cls)
        rng_state = d.pop("rng")
        o.__dict__.update(d)
        o.rng = np.random.default_rng()
        o.rng.bit_generator.state = rng_state
        return o
