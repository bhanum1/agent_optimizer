"""Round-robin multi-basin Thompson-sampling BO for a deterministic box with flat plateaus.

Regions are the top K distinct local optima (greedy over sorted y, separated by r_sep
units), recomputed every ask. Asks cycle through them. For the chosen region a
Matern-5/2 ARD GP is fit to the n_local points nearest its best point, y transformed to
log(y - min + shift), with a learned noise term (the plateaus act as noise). One joint
posterior sample is drawn over Gaussian perturbations (of a random subset of dims) of the
region's best at several scales, dropping candidates within excl units of any measured
point, and its argmin is asked. Local optima within skip_r of any `skip` point (saturated
basins) are never chosen as regions.
"""
import pickle
import warnings
import numpy as np
import torch


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, K=6, r_sep=1.2, scales=(0.05, 0.15, 0.4),
                 n_per=700, n_local=300, excl=0.04, shift=1.0, noise_max=0.3,
                 skip=(), skip_r=0.5):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lo)
        self.rng = np.random.default_rng(seed)
        self.K, self.r_sep, self.scales = K, r_sep, list(scales)
        self.n_per, self.n_local, self.excl = n_per, n_local, excl
        self.shift, self.noise_max = shift, noise_max
        self.skip = np.asarray(skip, float).reshape(-1, len(self.lo))
        self.skip_r = skip_r
        self.t = 0
        self.visits = []
        self.X = np.zeros((0, self.d))   # user units
        self.y = np.zeros(0)
        self.pending = []

    def _centers(self):
        o = np.argsort(self.y)
        reps = []
        for i in o:
            if len(self.skip) and np.min(np.linalg.norm(self.skip - self.X[i], axis=1)) < self.skip_r:
                continue
            if all(np.linalg.norm(self.X[i] - self.X[j]) > self.r_sep for j in reps):
                reps.append(i)
                if len(reps) >= self.K:
                    break
        return self.X[reps]

    def _pool(self, c):
        parts = []
        for s in self.scales:
            p = c + s * self.rng.standard_normal((self.n_per, self.d))
            mask = self.rng.random(p.shape) < 0.6
            mask[np.arange(self.n_per), self.rng.integers(self.d, size=self.n_per)] = True
            parts.append(np.where(mask, p, c))
        cand = np.clip(np.vstack(parts), self.lo, self.hi)
        near = self.X[np.linalg.norm(self.X - c, axis=1) < 2 * max(self.scales) + 1]
        if len(near):
            dmin = np.min(np.linalg.norm(cand[:, None] - near[None], axis=2), axis=1)
            if np.any(dmin > self.excl):
                cand = cand[dmin > self.excl]
        return cand

    def _fit_and_sample(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.constraints import Interval
        from gpytorch.likelihoods import GaussianLikelihood

        C = self._centers()
        c = C[self.t % len(C)]
        self.visits.append(c.tolist())
        self.t += 1
        Xt, yt = self.X, self.y
        if len(yt) > self.n_local:
            keep = np.argsort(np.linalg.norm(Xt - c, axis=1))[:self.n_local]
            Xt, yt = Xt[keep], yt[keep]
        z = np.log(yt - self.y.min() + self.shift)
        z = (z - z.mean()) / (z.std() + 1e-12)
        U = (Xt - self.lo) / (self.hi - self.lo)
        torch.manual_seed(int(self.rng.integers(2 ** 31)))
        tX = torch.tensor(U, dtype=torch.double)
        tz = torch.tensor(z, dtype=torch.double).unsqueeze(-1)
        lik = GaussianLikelihood(noise_constraint=Interval(1e-6, self.noise_max))
        kern = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.d,
                                        lengthscale_constraint=Interval(0.005, 4.0)),
                           outputscale_constraint=Interval(0.05, 20.0))
        gp = SingleTaskGP(tX, tz, covar_module=kern, likelihood=lik)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        cand = self._pool(c)
        gp.eval()
        with torch.no_grad():
            Uc = torch.tensor((cand - self.lo) / (self.hi - self.lo), dtype=torch.double)
            f = gp.posterior(Uc).rsample(torch.Size([1])).squeeze().numpy()
        return cand[int(np.argmin(f))]

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.y) < 2:
                x = self.lo + (self.hi - self.lo) * self.rng.random(self.d)
            else:
                x = self._fit_and_sample()
            x = x.tolist()
            self.pending.append(x)
            out.append(x)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            if x in self.pending:
                self.pending.remove(x)
            self.X = np.vstack([self.X, np.asarray(x, float)])
            self.y = np.append(self.y, float(v))

    def save(self, path):
        d = dict(self.__dict__)
        d["rng"] = self.rng.bit_generator.state
        with open(path, "wb") as fh:
            pickle.dump(d, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        obj = cls.__new__(cls)
        rs = d.pop("rng")
        obj.__dict__.update(d)
        obj.rng = np.random.default_rng()
        obj.rng.bit_generator.state = rs
        return obj
