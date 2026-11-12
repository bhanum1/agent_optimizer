"""TuRBO-1 style trust-region Bayesian optimization (deterministic, unconstrained).

GP (Matern-5/2 ARD) on a log-shifted, standardized objective, fitted on the points
nearest the trust-region center; Thompson sampling over Sobol candidates inside an
ARD-shaped trust region. On collapse the region restarts from a fresh Sobol design.
"""
import json
import math

import numpy as np
import torch
from scipy.stats import qmc


class Optimizer:
    max_ask = 1
    batch = False

    def __init__(self, space, seed=0, n_init=10, length_init=0.8, length_min=0.0078125,
                 length_max=1.6, succ_tol=3, fail_tol=5, n_cand=4096, max_fit=250):
        self.lo = [b[0] for b in space["bounds"]]
        self.hi = [b[1] for b in space["bounds"]]
        self.d = len(self.lo)
        self.cfg = dict(seed=seed, n_init=n_init, length_init=length_init,
                        length_min=length_min, length_max=length_max, succ_tol=succ_tol,
                        fail_tol=fail_tol, n_cand=n_cand, max_fit=max_fit)
        self.rng = np.random.default_rng(seed)
        self.X = []          # unit-cube points, all observations
        self.y = []
        self.run = []        # indices of observations in the current TR run
        self.init_queue = []  # pending design points of the current run (unit cube)
        self.pending = []
        self.length = length_init
        self.succ = 0
        self.fail = 0
        self.n_restarts = 0
        self.run_best_hist = []

    # ---------- helpers ----------
    def _to_unit(self, x):
        return [(v - l) / (h - l) for v, l, h in zip(x, self.lo, self.hi)]

    def _from_unit(self, u):
        return [float(min(h, max(l, l + v * (h - l)))) for v, l, h in zip(u, self.lo, self.hi)]

    def _new_run(self):
        if self.run:
            self.run_best_hist.append(float(min(self.y[i] for i in self.run)))
        self.n_restarts += 1
        self.run = []
        self.length = self.cfg["length_init"]
        self.succ = 0
        self.fail = 0
        s = qmc.Sobol(self.d, scramble=True, seed=int(self.rng.integers(2**31)))
        self.init_queue = s.random(self.cfg["n_init"]).tolist()

    def _transform(self, y):
        y = np.asarray(y, float)
        ymin = y.min()
        eps = max(1e-9, 0.01 * (np.median(y) - ymin))
        z = np.log(y - ymin + eps)
        return (z - z.mean()) / (z.std() + 1e-12)

    def _fit(self, center):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.constraints import Interval
        from gpytorch.likelihoods import GaussianLikelihood

        X = np.asarray(self.X)
        y = np.asarray(self.y)
        if len(X) > self.cfg["max_fit"]:
            dist = np.linalg.norm(X - center, axis=1)
            idx = np.argsort(dist)[: self.cfg["max_fit"]]
            X, y = X[idx], y[idx]
        z = self._transform(y)
        tX = torch.tensor(X, dtype=torch.float64)
        tz = torch.tensor(z, dtype=torch.float64).unsqueeze(-1)
        lik = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-2))
        kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.d,
                                        lengthscale_constraint=Interval(0.005, 4.0)),
                           outputscale_constraint=Interval(0.05, 20.0))
        model = SingleTaskGP(tX, tz, likelihood=lik, covar_module=kern)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        return model

    def _ts_point(self, taken):
        X = np.asarray(self.X)
        run_idx = self.run
        ry = np.asarray([self.y[i] for i in run_idx])
        center = X[run_idx[int(np.argmin(ry))]]
        model = self._fit(center)
        ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        w = ls / ls.mean()
        w = w / np.prod(w) ** (1.0 / self.d)
        lb = np.clip(center - w * self.length / 2, 0, 1)
        ub = np.clip(center + w * self.length / 2, 0, 1)
        s = qmc.Sobol(self.d, scramble=True, seed=int(self.rng.integers(2**31)))
        pert = lb + (ub - lb) * s.random(self.cfg["n_cand"])
        prob = min(20.0 / self.d, 1.0)
        mask = self.rng.random(pert.shape) <= prob
        empty = ~mask.any(axis=1)
        mask[empty, self.rng.integers(0, self.d, size=empty.sum())] = True
        cand = np.where(mask, pert, center)
        torch.manual_seed(int(self.rng.integers(2**31)))
        with torch.no_grad():
            post = model.posterior(torch.tensor(cand, dtype=torch.float64))
            f = post.rsample(torch.Size([1])).squeeze().numpy()
        order = np.argsort(f)
        for i in order:
            c = cand[i]
            if all(np.max(np.abs(c - t)) > 1e-9 for t in taken):
                return c.tolist()
        return cand[order[0]].tolist()

    # ---------- interface ----------
    def ask(self, n):
        out = []
        for _ in range(n):
            if not self.run and not self.init_queue:
                self._new_run()
            if self.init_queue:
                u = self.init_queue.pop(0)
            else:
                taken = [np.asarray(p) for p in self.pending] + [np.asarray(p) for p in out]
                u = self._ts_point(taken)
            out.append(u)
            self.pending.append(u)
        return [self._from_unit(u) for u in out]

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            u = self._to_unit(x)
            v = float(v)
            if not math.isfinite(v):
                v = max(self.y) * 2 + 1 if self.y else 1e30
            # remove matching pending; points never asked (warm start) only join the run
            asked = False
            for k, p in enumerate(self.pending):
                if np.max(np.abs(np.asarray(p) - np.asarray(u))) < 1e-9:
                    self.pending.pop(k)
                    asked = True
                    break
            in_init = len(self.init_queue) > 0 or len(self.run) < self.cfg["n_init"]
            prev_best = min((self.y[i] for i in self.run), default=math.inf)
            self.X.append(u)
            self.y.append(v)
            self.run.append(len(self.y) - 1)
            if not asked or in_init or not math.isfinite(prev_best):
                continue
            if v < prev_best - 1e-3 * abs(prev_best):
                self.succ += 1
                self.fail = 0
            else:
                self.succ = 0
                self.fail += 1
            if self.succ >= self.cfg["succ_tol"]:
                self.length = min(2 * self.length, self.cfg["length_max"])
                self.succ = 0
            elif self.fail >= self.cfg["fail_tol"]:
                self.length /= 2
                self.fail = 0
            if self.length < self.cfg["length_min"]:
                self.run_best_hist.append(float(min(self.y[i] for i in self.run)))
                self.run = []
                self.init_queue = []

    def save(self, path):
        st = dict(lo=self.lo, hi=self.hi, d=self.d, cfg=self.cfg,
                  rng=self.rng.bit_generator.state, X=self.X, y=self.y, run=self.run,
                  init_queue=self.init_queue, pending=self.pending, length=self.length,
                  succ=self.succ, fail=self.fail, n_restarts=self.n_restarts,
                  run_best_hist=self.run_best_hist)
        with open(path, "w") as f:
            json.dump(st, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            st = json.load(f)
        self = cls.__new__(cls)
        rng_state = st.pop("rng")
        for k, v in st.items():
            setattr(self, k, v)
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = rng_state
        return self
