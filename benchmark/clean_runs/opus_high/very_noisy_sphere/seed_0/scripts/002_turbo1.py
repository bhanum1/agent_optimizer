"""TuRBO-1 style trust-region BO for a noisy continuous box.

GP (Matern 5/2, ARD) on log-shifted standardized y; LogEI against the best
posterior mean, maximized inside a trust region centred on the observed point
with the lowest posterior mean. The region expands after `succ_tol` successes
and shrinks after `fail_tol` failures; when it gets below `length_min` it is
reset to `length_init` and recentred (data kept). A fraction `p_global` of
asks optimize LogEI over the whole box instead.
"""
import json
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, length_init=0.4, length_min=0.02,
                 length_max=0.8, succ_tol=3, fail_tol=None, p_global=0.1,
                 n_init=10, max_fit=600):
        self.lb = [float(b[0]) for b in space["bounds"]]
        self.ub = [float(b[1]) for b in space["bounds"]]
        self.d = len(self.lb)
        self.seed = int(seed)
        self.length = float(length_init)
        self.length_init = float(length_init)
        self.length_min = float(length_min)
        self.length_max = float(length_max)
        self.succ_tol = int(succ_tol)
        self.fail_tol = int(fail_tol) if fail_tol else max(4, self.d)
        self.p_global = float(p_global)
        self.n_init = int(n_init)
        self.max_fit = int(max_fit)
        self.X, self.y = [], []
        self.pending = []
        self.succ = 0
        self.fail = 0
        self.n_ask = 0
        self.n_restart = 0
        self.rng_state = np.random.default_rng(self.seed).bit_generator.state

    # ---- helpers
    def _rng(self):
        r = np.random.default_rng()
        r.bit_generator.state = self.rng_state
        return r

    def _to_unit(self, X):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return (np.asarray(X, float) - lb) / (ub - lb)

    def _from_unit(self, U):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return lb + np.clip(U, 0, 1) * (ub - lb)

    def _fit(self):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        torch.manual_seed(self.seed + self.n_ask)
        X = self._to_unit(self.X)
        y = np.asarray(self.y, float)
        if len(y) > self.max_fit:  # keep the best half and the most recent
            keep = set(np.argsort(y)[: self.max_fit // 2].tolist())
            keep |= set(range(len(y) - self.max_fit // 2, len(y)))
            idx = np.array(sorted(keep))
            X, y = X[idx], y[idx]
        spread = max(np.std(y), 1e-9)
        z = np.log(y - y.min() + 0.1 * spread)
        z = (z - z.mean()) / max(z.std(), 1e-9)
        Xt = torch.tensor(X, dtype=torch.double)
        Yt = torch.tensor(z, dtype=torch.double).unsqueeze(-1)
        gp = SingleTaskGP(Xt, Yt)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        return gp

    # ---- interface
    def ask(self, n):
        out = []
        for _ in range(n):
            out.append(self._ask_one())
        return out

    def _ask_one(self):
        rng = self._rng()
        self.n_ask += 1
        if len(self.y) < self.n_init:
            x = self._from_unit(rng.random(self.d)).tolist()
            self.rng_state = rng.bit_generator.state
            self.pending.append(x)
            return x
        import torch
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        gp = self._fit()
        Xu = torch.tensor(self._to_unit(self.X), dtype=torch.double)
        with torch.no_grad():
            mu = gp.posterior(Xu).mean.squeeze(-1)
        ib = int(torch.argmin(mu))
        best_f = float(mu[ib])
        center = Xu[ib].numpy()
        if rng.random() < self.p_global:
            lo, hi = np.zeros(self.d), np.ones(self.d)
        else:
            try:
                ls = gp.covar_module.base_kernel.lengthscale
            except AttributeError:
                ls = gp.covar_module.lengthscale
            w = ls.detach().numpy().ravel()
            w = w / np.prod(w) ** (1.0 / self.d)
            w = np.clip(w, 0.25, 4.0)
            half = self.length * w / 2
            lo = np.clip(center - half, 0, 1)
            hi = np.clip(center + half, 0, 1)
        acq = LogExpectedImprovement(gp, best_f=best_f, maximize=False)
        bounds = torch.tensor(np.vstack([lo, hi]), dtype=torch.double)
        torch.manual_seed(int(rng.integers(1 << 30)))
        try:
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=8,
                                    raw_samples=256)
            u = cand.detach().numpy().ravel()
        except Exception:
            u = lo + rng.random(self.d) * (hi - lo)
        x = self._from_unit(u).tolist()
        self.rng_state = rng.bit_generator.state
        self.pending.append(x)
        return x

    def tell(self, X, y, g=None):
        for xi, yi in zip(X, y):
            xi = [float(v) for v in xi]
            yi = float(yi)
            if not math.isfinite(yi):
                yi = max(self.y) if self.y else 1e9
            mine = False
            for k, p in enumerate(self.pending):
                if np.allclose(p, xi, atol=1e-9):
                    self.pending.pop(k)
                    mine = True
                    break
            prev = min(self.y) if self.y else math.inf
            self.X.append(xi)
            self.y.append(yi)
            if mine and len(self.y) > self.n_init:
                if yi < prev - 1e-3 * abs(prev):
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
                    self.n_restart += 1

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls.__new__(cls)
        o.__dict__.update(d)
        return o
