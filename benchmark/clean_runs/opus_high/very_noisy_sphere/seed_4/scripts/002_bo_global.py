"""GP Bayesian optimization on a log-warped objective, noise-aware (LogNEI).

Optionally starts by re-measuring the best `n_rep` already-told points (noise estimate),
and can restrict search to a trust box around the best posterior-mean point (`tr_frac`).
"""
import json
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_rep=0, tr_frac=None, raw_samples=1024,
                 num_restarts=10, max_train=800):
        b = np.array(space["bounds"], dtype=float)
        self.lo, self.hi = b[:, 0].tolist(), b[:, 1].tolist()
        self.seed = int(seed)
        self.n_rep = int(n_rep)
        self.tr_frac = tr_frac
        self.raw_samples = int(raw_samples)
        self.num_restarts = int(num_restarts)
        self.max_train = int(max_train)
        self.X, self.y = [], []
        self.rep_queue = None
        self.n_asked = 0

    # ---- helpers ----
    def _unit(self, X):
        lo, hi = np.array(self.lo), np.array(self.hi)
        return (np.asarray(X, float) - lo) / (hi - lo)

    def _from_unit(self, U):
        lo, hi = np.array(self.lo), np.array(self.hi)
        return lo + np.clip(U, 0, 1) * (hi - lo)

    def _warp(self, y):
        y = np.asarray(y, float)
        q25, q50 = np.percentile(y, [25, 50])
        s = max(q50 - q25, 1e-9)
        z = np.log(y - y.min() + s)
        return (z - z.mean()) / (z.std() + 1e-12)

    def _fit(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        X = np.array(self.X); y = np.array(self.y)
        if len(y) > self.max_train:
            # keep the best half of the budget plus a random rest
            order = np.argsort(y)
            keep = list(order[: self.max_train // 2])
            rng = np.random.default_rng(self.seed + len(y))
            rest = order[self.max_train // 2:]
            keep += list(rng.choice(rest, self.max_train - len(keep), replace=False))
            X, y = X[keep], y[keep]
        z = self._warp(y)
        Xt = torch.tensor(self._unit(X)); zt = torch.tensor(z).unsqueeze(-1)
        m = SingleTaskGP(Xt, zt)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
        except Exception:
            pass
        return m, Xt

    def _next_bo(self):
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.optim import optimize_acqf
        torch.manual_seed(self.seed * 100003 + self.n_asked)
        m, Xt = self._fit()
        d = Xt.shape[1]
        bounds = torch.stack([torch.zeros(d), torch.ones(d)])
        if self.tr_frac:
            with torch.no_grad():
                mu = m.posterior(Xt).mean.squeeze(-1)
            c = Xt[int(torch.argmin(mu))]
            ls = m.covar_module.lengthscale.detach().squeeze()
            w = ls / ls.prod() ** (1.0 / d) * float(self.tr_frac)
            bounds = torch.stack([(c - w / 2).clamp(0, 1), (c + w / 2).clamp(0, 1)])
        # minimize: botorch maximizes, so negate the model output via objective
        from botorch.acquisition.objective import LinearMCObjective
        obj = LinearMCObjective(torch.tensor([-1.0]))
        acq = qLogNoisyExpectedImprovement(m, X_baseline=Xt, objective=obj, prune_baseline=True)
        cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=self.num_restarts,
                                raw_samples=self.raw_samples)
        return self._from_unit(cand.detach().numpy()[0]).tolist()

    # ---- interface ----
    def ask(self, n):
        out = []
        for _ in range(n):
            if self.rep_queue is None:
                order = np.argsort(self.y) if self.y else []
                self.rep_queue = [list(self.X[i]) for i in order[: self.n_rep]]
            if self.rep_queue:
                p = self.rep_queue.pop(0)
            elif len(self.y) < 2:
                rng = np.random.default_rng(self.seed + self.n_asked)
                p = self._from_unit(rng.random(len(self.lo))).tolist()
            else:
                p = self._next_bo()
            self.n_asked += 1
            out.append([float(v) for v in p])
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append([float(v) for v in xi]); self.y.append(float(yi))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path) as f:
            o.__dict__.update(json.load(f))
        return o
