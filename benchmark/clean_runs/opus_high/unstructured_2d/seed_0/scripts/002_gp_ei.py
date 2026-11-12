import pickle
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimizer:
    """GP (Matern-5/2, ARD) + LogEI, sequential, deterministic objective."""

    def __init__(self, space, seed=0, n_init=0, xi_explore=0.0):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.X, self.y = [], []
        self.pending = []
        self.n_asked = 0
        self.rng = np.random.default_rng(self.seed)

    def _random(self):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return list(lo + self.rng.random(len(lo)) * (hi - lo))

    def _propose(self):
        if len(self.y) < max(self.n_init, 3):
            return self._random()
        torch.manual_seed(self.seed + self.n_asked)
        X = torch.tensor(np.array(self.X), dtype=torch.double)
        y = np.array(self.y, dtype=float)
        # rank-preserving squash of large values keeps the GP focused on the low region
        med = np.median(y)
        yt = np.where(y > med, med + np.log1p(y - med), y)
        Y = torch.tensor(yt, dtype=torch.double).unsqueeze(-1)
        b = torch.tensor(self.bounds.T, dtype=torch.double)
        model = SingleTaskGP(X, -Y, input_transform=Normalize(d=X.shape[-1], bounds=b),
                             outcome_transform=Standardize(m=1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        acq = LogExpectedImprovement(model, best_f=float((-Y).max()))
        cand, _ = optimize_acqf(acq, bounds=b, q=1, num_restarts=20, raw_samples=1024)
        x = cand.detach().numpy()[0]
        x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        # avoid exact repeats (deterministic objective)
        Xa = np.array(self.X)
        span = self.bounds[:, 1] - self.bounds[:, 0]
        if np.min(np.max(np.abs(Xa - x) / span, axis=1)) < 1e-4:
            return self._random()
        return list(map(float, x))

    def ask(self, n):
        out = []
        for _ in range(n):
            x = self._propose()
            self.n_asked += 1
            self.pending.append(x)
            out.append(x)
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            xi = list(map(float, xi))
            self.X.append(xi)
            self.y.append(float(yi))
            for j, p in enumerate(self.pending):
                if np.allclose(p, xi):
                    self.pending.pop(j)
                    break

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
