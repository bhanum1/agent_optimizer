"""Round-robin local Thompson sampling in several fixed boxes (other basins).

`centers` are points in the encoded space (user units here); each box is
center +- `half` clipped to bounds. Asks cycle through the boxes. In a box, a GP
is fitted to rank-warped y (normal scores) of all observations within
center +- 2*half, one joint posterior draw is taken over n_cand Sobol points in
the box plus perturbations of the best local observations, and its minimizer is
returned. With fewer than `min_fit` local points, a uniform point is returned.
"""
import pickle
import numpy as np
import torch
from scipy.stats import norm, qmc


class Optimizer:
    max_ask = 1

    def __init__(self, space, centers, half=1.5, seed=0, n_cand=1500, n_local=1000,
                 local_sd=0.1, min_fit=6):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.dim = space["dim"]
        self.centers = np.array(centers, dtype=float)
        self.half = float(half)
        self.rng = np.random.default_rng(seed)
        self.n_cand, self.n_local, self.local_sd, self.min_fit = n_cand, n_local, local_sd, min_fit
        self.k = 0
        self.X, self.y = [], []

    def _box(self, c, h):
        lo = np.maximum(c - h, self.bounds[:, 0])
        hi = np.minimum(c + h, self.bounds[:, 1])
        return lo, hi

    def _one(self, c):
        lo, hi = self._box(c, self.half)
        flo, fhi = self._box(c, 2 * self.half)
        X = np.array(self.X).reshape(-1, self.dim)
        y = np.array(self.y)
        inside = np.all((X >= flo) & (X <= fhi), axis=1) if len(X) else np.zeros(0, bool)
        if inside.sum() < self.min_fit:
            return lo + self.rng.random(self.dim) * (hi - lo)
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        Xl = (X[inside] - flo) / (fhi - flo)
        yl = y[inside]
        r = np.argsort(np.argsort(yl))
        z = norm.ppf((r + 0.5) / len(yl))
        z = (z - z.mean()) / z.std()
        torch.manual_seed(int(self.rng.integers(2**31)))
        gp = SingleTaskGP(torch.tensor(Xl, dtype=torch.double),
                          torch.tensor(z, dtype=torch.double).unsqueeze(-1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        ulo, uhi = (lo - flo) / (fhi - flo), (hi - flo) / (fhi - flo)
        cg = ulo + qmc.Sobol(self.dim, seed=int(self.rng.integers(2**31))).random(self.n_cand) * (uhi - ulo)
        top = Xl[np.argsort(z)[:5]]
        cl = top[self.rng.integers(len(top), size=self.n_local)]
        cl = np.clip(cl + self.local_sd * self.rng.standard_normal(cl.shape), ulo, uhi)
        C = torch.tensor(np.vstack([cg, cl]), dtype=torch.double)
        with torch.no_grad():
            f = gp.posterior(C).rsample(torch.Size([1])).squeeze()
        u = C[int(torch.argmin(f))].numpy()
        return np.clip(flo + u * (fhi - flo), lo, hi)

    def ask(self, n):
        out = []
        for _ in range(n):
            c = self.centers[self.k % len(self.centers)]
            self.k += 1
            out.append([float(v) for v in self._one(c)])
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
