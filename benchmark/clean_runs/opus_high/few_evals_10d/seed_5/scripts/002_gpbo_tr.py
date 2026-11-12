"""GP Bayesian optimization with an optional TuRBO-style trust region.

GP (SingleTaskGP, ARD Matern-5/2, learned noise) on inputs scaled to [0,1] and
standardized y. Acquisition: qLogNoisyExpectedImprovement, optimized inside a
trust region centred on the observed point with the best posterior mean (tr=True),
or over the whole box (tr=False). TR side length doubles after `succ_tol`
consecutive improvements and halves after `fail_tol` consecutive non-improvements;
when it shrinks below `L_min` it resets to `L_init`.
"""
import pickle
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, tr=True, L_init=0.8, L_min=0.05, L_max=1.6,
                 succ_tol=3, fail_tol=6, raw_samples=1024, restarts=10):
        b = np.array(space["bounds"], float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.dim = len(b)
        self.seed = seed
        self.tr = tr
        self.L = L_init
        self.L_init, self.L_min, self.L_max = L_init, L_min, L_max
        self.succ_tol, self.fail_tol = succ_tol, fail_tol
        self.raw_samples, self.restarts = raw_samples, restarts
        self.X, self.y = [], []
        self.n_ask = 0
        self.succ = self.fail = 0
        self.best_seen = np.inf
        self.history = []

    # --- helpers
    def _unit(self, X):
        return (np.asarray(X, float) - self.lo) / (self.hi - self.lo)

    def _fit(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        Xt = torch.tensor(self._unit(self.X))
        y = np.array(self.y)
        yt = torch.tensor((y - y.mean()) / (y.std() + 1e-12)).unsqueeze(-1)
        m = SingleTaskGP(Xt, yt)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
        except Exception:
            pass
        return m, Xt

    def ask(self, n):
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.optim import optimize_acqf
        torch.manual_seed(self.seed * 100003 + self.n_ask)
        self.n_ask += 1
        out = []
        for _ in range(n):
            m, Xt = self._fit()
            with torch.no_grad():
                mu = m.posterior(Xt).mean.squeeze(-1)
            centre = Xt[int(mu.argmin())]
            from botorch.acquisition.objective import GenericMCObjective
            obj = GenericMCObjective(lambda Z, X=None: -Z[..., 0])
            acq = qLogNoisyExpectedImprovement(m, X_baseline=Xt, objective=obj, prune_baseline=True)
            if self.tr:
                try:
                    ls = m.covar_module.lengthscale
                except AttributeError:
                    ls = m.covar_module.base_kernel.lengthscale
                w = ls.detach().squeeze().numpy()
                w = w / np.prod(w) ** (1.0 / self.dim)
                half = self.L * w / 2
                lb = torch.tensor(np.clip(centre.numpy() - half, 0, 1))
                ub = torch.tensor(np.clip(centre.numpy() + half, 0, 1))
            else:
                lb, ub = torch.zeros(self.dim), torch.ones(self.dim)
            cand, _ = optimize_acqf(acq, bounds=torch.stack([lb, ub]), q=1,
                                    num_restarts=self.restarts, raw_samples=self.raw_samples)
            x = cand[0].detach().numpy()
            x = np.clip(self.lo + x * (self.hi - self.lo), self.lo, self.hi)
            out.append([float(v) for v in x])
        return out

    def tell(self, X, y, g):
        if not len(y):
            return
        warm = len(y) > 1
        for x, v in zip(X, y):
            v = float(v)
            if not warm and self.X:
                # TR bookkeeping on our own sequential points
                if v < self.best_seen - 1e-3 * abs(self.best_seen):
                    self.succ += 1
                    self.fail = 0
                else:
                    self.fail += 1
                    self.succ = 0
                if self.succ >= self.succ_tol:
                    self.L = min(2 * self.L, self.L_max)
                    self.succ = 0
                elif self.fail >= self.fail_tol:
                    self.L /= 2
                    self.fail = 0
                    if self.L < self.L_min:
                        self.L = self.L_init
                self.history.append(self.L)
            self.X.append([float(a) for a in x])
            self.y.append(v)
            self.best_seen = min(self.best_seen, v)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
