"""GP Thompson sampling for a noisy objective with heavy-tailed values.

y is warped to normal scores of its ranks (copula) before fitting, as in gpbo.py.
Each ask draws one joint posterior sample over a candidate set (n_global Sobol
points over the whole box plus n_local Gaussian perturbations, sd `local_sd` in
the unit cube, of the `n_top` best posterior-mean observed points) and returns
its minimizer. Thompson sampling keeps exploring other basins in proportion to
their posterior probability of holding the minimum.
"""
import pickle
import numpy as np
import torch
from scipy.stats import norm, qmc


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_global=2048, n_local=1500, local_sd=0.05, n_top=10):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.dim = space["dim"]
        self.rng = np.random.default_rng(seed)
        self.n_global = n_global
        self.n_local = n_local
        self.local_sd = local_sd
        self.n_top = n_top
        self.X, self.y = [], []

    def _fit(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        X = (np.array(self.X) - lo) / (hi - lo)
        y = np.array(self.y)
        r = np.argsort(np.argsort(y))
        z = norm.ppf((r + 0.5) / len(y))
        z = (z - z.mean()) / z.std()
        Xt = torch.tensor(X, dtype=torch.double)
        zt = torch.tensor(z, dtype=torch.double).unsqueeze(-1)
        gp = SingleTaskGP(Xt, zt)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        return gp, Xt

    def ask(self, n):
        out = []
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        for _ in range(n):
            torch.manual_seed(int(self.rng.integers(2**31)))
            gp, Xt = self._fit()
            with torch.no_grad():
                mu = gp.posterior(Xt).mean.squeeze(-1).numpy()
            top = Xt.numpy()[np.argsort(mu)[:self.n_top]]
            cg = qmc.Sobol(self.dim, seed=int(self.rng.integers(2**31))).random(self.n_global)
            cl = top[self.rng.integers(len(top), size=self.n_local)]
            cl = (cl + self.local_sd * self.rng.standard_normal(cl.shape)).clip(0, 1)
            C = torch.tensor(np.vstack([cg, cl]), dtype=torch.double)
            with torch.no_grad():
                f = gp.posterior(C).rsample(torch.Size([1])).squeeze()
            u = C[int(torch.argmin(f))].numpy()
            out.append([float(v) for v in lo + u * (hi - lo)])
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
