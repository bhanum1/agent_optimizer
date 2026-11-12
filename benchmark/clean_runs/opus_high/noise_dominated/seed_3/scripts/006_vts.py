"""Local GP Thompson sampling inside a fixed box (the valley), on a log-warped mean.

Only observations inside the box (widened by `margin` user units) are modelled.
y is clipped above at `clip`, then warped to log(y - L) with L = min(y) - `shift`,
so the GP spends its resolution on the low end. With noise that is skewed upward and
grows with y, the posterior mean is lowest where the floor is lowest.
Each ask draws one joint posterior sample over n_cand Sobol points in the box plus
n_local Gaussian perturbations (sd `local_sd`, unit cube of the box) of the
`n_top` best posterior-mean observed points, and returns the draw's minimizer.
"""
import pickle
import numpy as np
import torch
from scipy.stats import qmc


class Optimizer:
    max_ask = 1

    def __init__(self, space, lo, hi, seed=0, margin=0.5, clip=-120.0, shift=1.5,
                 n_cand=1500, n_local=1500, local_sd=0.05, n_top=10):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.dim = space["dim"]
        self.lo = np.maximum(np.array(lo, dtype=float), self.bounds[:, 0])
        self.hi = np.minimum(np.array(hi, dtype=float), self.bounds[:, 1])
        self.rng = np.random.default_rng(seed)
        self.margin = margin
        self.clip = clip
        self.shift = shift
        self.n_cand = n_cand
        self.n_local = n_local
        self.local_sd = local_sd
        self.n_top = n_top
        self.X, self.y = [], []

    def _fit(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        X = np.array(self.X)
        y = np.array(self.y)
        w = np.all((X >= self.lo - self.margin) & (X <= self.hi + self.margin), axis=1)
        X, y = X[w], np.minimum(y[w], self.clip)
        t = np.log(y - (y.min() - self.shift))
        t = (t - t.mean()) / t.std()
        Xt = torch.tensor((X - self.lo) / (self.hi - self.lo), dtype=torch.double)
        tt = torch.tensor(t, dtype=torch.double).unsqueeze(-1)
        gp = SingleTaskGP(Xt, tt)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        return gp, Xt

    def ask(self, n):
        out = []
        for _ in range(n):
            torch.manual_seed(int(self.rng.integers(2**31)))
            X = np.array(self.X)
            w = np.all((X >= self.lo - self.margin) & (X <= self.hi + self.margin), axis=1) if len(X) else []
            if len(X) == 0 or w.sum() < 10:
                u = self.rng.random(self.dim)
            else:
                gp, Xt = self._fit()
                with torch.no_grad():
                    mu = gp.posterior(Xt).mean.squeeze(-1).numpy()
                top = Xt.numpy()[np.argsort(mu)[:self.n_top]].clip(0, 1)
                cg = qmc.Sobol(self.dim, seed=int(self.rng.integers(2**31))).random(self.n_cand)
                cl = top[self.rng.integers(len(top), size=self.n_local)]
                cl = (cl + self.local_sd * self.rng.standard_normal(cl.shape)).clip(0, 1)
                C = torch.tensor(np.vstack([cg, cl]), dtype=torch.double)
                with torch.no_grad():
                    f = gp.posterior(C).rsample(torch.Size([1])).squeeze()
                u = C[int(torch.argmin(f))].numpy()
            out.append([float(v) for v in self.lo + u * (self.hi - self.lo)])
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append([float(v) for v in xi])
            self.y.append(float(yi))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
