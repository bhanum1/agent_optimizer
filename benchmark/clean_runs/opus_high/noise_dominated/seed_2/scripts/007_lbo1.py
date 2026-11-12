import pickle
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


class Optimizer:
    """Local GP-BO: like robo.py (Matern-5/2 ARD, learned noise, upward-outlier
    trimming via LOO residuals, qLogNoisyEI), but restricted to the box
    center +- half (clipped to bounds); only data inside that box is modelled."""

    def __init__(self, space, center, half, seed=0, z_out=3.0, refit_every=10,
                 refit_after=0, num_restarts=8, raw_samples=512):
        blo = np.array([b[0] for b in space["bounds"]], float)
        bhi = np.array([b[1] for b in space["bounds"]], float)
        c = np.asarray(center, float)
        h = np.broadcast_to(np.asarray(half, float), c.shape)
        self.lo = np.maximum(blo, c - h)
        self.hi = np.minimum(bhi, c + h)
        self.z_out = z_out
        self.refit_every = refit_every
        self.refit_after = refit_after
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.X, self.y = [], []
        self.hyper = None
        self.n_since_fit = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _u(self, X):
        return (np.asarray(X) - self.lo) / (self.hi - self.lo)

    def _fit(self, U, Y, refit):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.models.transforms import Standardize
        tx = torch.tensor(U)
        ty = torch.tensor(Y).unsqueeze(-1)
        model = SingleTaskGP(tx, ty, outcome_transform=Standardize(m=1))
        if self.hyper is not None:
            try:
                model.load_state_dict(self._hyper_dict(model), strict=False)
            except Exception:
                pass
        if refit or self.hyper is None:
            try:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            except Exception:
                pass
            self.hyper = {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()
                          if "raw" in k}
        model.eval()
        return model

    def _hyper_dict(self, model):
        sd = model.state_dict()
        out = {}
        for k, v in self.hyper.items():
            if k in sd:
                out[k] = torch.tensor(v).reshape(sd[k].shape)
        return out

    def _inliers(self, model, U, Y):
        # LOO residuals in the model's standardized space; flag large positive ones
        with torch.no_grad():
            tx = torch.tensor(U)
            K = model.covar_module(tx).to_dense()
            noise = model.likelihood.noise.detach().reshape(-1)[0]
            K = K + noise * torch.eye(len(U))
            ys = model.outcome_transform(torch.tensor(Y).unsqueeze(-1))[0].squeeze(-1)
            mean = model.mean_module(tx)
            Ki = torch.linalg.inv(K)
            alpha = Ki @ (ys - mean)
            d = torch.diagonal(Ki)
            r = alpha / d
            z = (r * torch.sqrt(d)).numpy()
        return z < self.z_out

    def ask(self, n):
        out = []
        for _ in range(n):
            out.append(self._ask_one())
        return out

    def _ask_one(self):
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.optim import optimize_acqf
        d = len(self.lo)
        if len(self.y) < 2 * d:
            return list(self.lo + self.rng.random(d) * (self.hi - self.lo))
        U = self._u(self.X)
        Y = np.asarray(self.y, float)
        n = len(Y)
        refit = n < self.refit_after or self.n_since_fit >= self.refit_every
        model = self._fit(U, Y, refit)
        keep = self._inliers(model, U, Y)
        if (~keep).any() and keep.sum() >= 2 * d:
            model = self._fit(U[keep], Y[keep], refit)
            keep2 = self._inliers(model, U[keep], Y[keep])
            idx = np.where(keep)[0][keep2]
            if len(idx) < keep.sum() and len(idx) >= 2 * d:
                model = self._fit(U[idx], Y[idx], refit)
                keep = np.zeros(n, bool)
                keep[idx] = True
        self.n_since_fit = 0 if refit else self.n_since_fit + 1
        torch.manual_seed(int(self.rng.integers(2**31)))
        acq = qLogNoisyExpectedImprovement(model, X_baseline=torch.tensor(U[keep]),
                                           prune_baseline=True)
        b = torch.stack([torch.zeros(d), torch.ones(d)])
        try:
            cand, _ = optimize_acqf(acq, b, q=1, num_restarts=self.num_restarts,
                                    raw_samples=self.raw_samples)
            u = cand.detach().numpy()[0]
        except Exception:
            u = self.rng.random(d)
        u = np.clip(u, 0, 1)
        return list(self.lo + u * (self.hi - self.lo))

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v)
            if not np.isfinite(v):
                v = max(self.y) if self.y else 1e6
            x = np.asarray(x, float)
            if np.any(x < self.lo - 1e-9) or np.any(x > self.hi + 1e-9):
                continue
            self.X.append([float(a) for a in x])
            self.y.append(v)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
