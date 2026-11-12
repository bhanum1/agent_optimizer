import json
import numpy as np
import torch


class Optimizer:
    """Noise-aware GP BO: SingleTaskGP (Matern-5/2 ARD) + qLogNEI, optional trust region."""
    max_ask = 1

    def __init__(self, space, seed=0, n_init=10, tr_frac=None, raw_samples=1024, restarts=10):
        self.bounds = [list(map(float, b)) for b in space["bounds"]]
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.tr_frac = tr_frac
        self.raw_samples = int(raw_samples)
        self.restarts = int(restarts)
        self.X, self.y, self.pending = [], [], []
        self.calls = 0

    def _rand(self, n):
        rng = np.random.default_rng(self.seed * 100003 + self.calls)
        lo, hi = np.array(self.bounds).T
        return (lo + rng.random((n, len(lo))) * (hi - lo)).tolist()

    def ask(self, n):
        self.calls += 1
        if len(self.X) < self.n_init:
            pts = self._rand(n)
        else:
            pts = [self._bo_point() for _ in range(n)]
        self.pending += pts
        return pts

    def _bo_point(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.optim import optimize_acqf
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        torch.manual_seed(self.seed * 7919 + self.calls)
        dt = torch.double
        b = torch.tensor(self.bounds, dtype=dt).T
        X = torch.tensor(self.X, dtype=dt)
        Y = -torch.tensor(self.y, dtype=dt).unsqueeze(-1)  # botorch maximizes
        model = SingleTaskGP(X, Y, input_transform=Normalize(X.shape[-1], bounds=b),
                             outcome_transform=Standardize(1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        Xp = torch.tensor(self.pending, dtype=dt) if self.pending else None
        acq = qLogNoisyExpectedImprovement(model, X_baseline=X, X_pending=Xp, prune_baseline=True)
        ob = b
        if self.tr_frac:
            with torch.no_grad():
                mu = model.posterior(X).mean.squeeze(-1)
            c = X[int(torch.argmax(mu))]
            ls = model.covar_module.lengthscale.detach().squeeze() if hasattr(model.covar_module, "lengthscale") \
                else model.covar_module.base_kernel.lengthscale.detach().squeeze()
            w = ls / ls.prod().pow(1.0 / len(ls))
            half = 0.5 * float(self.tr_frac) * w * (b[1] - b[0])
            ob = torch.stack([torch.maximum(c - half, b[0]), torch.minimum(c + half, b[1])])
        cand, _ = optimize_acqf(acq, bounds=ob, q=1, num_restarts=self.restarts,
                                raw_samples=self.raw_samples)
        return cand.squeeze(0).tolist()

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = [float(t) for t in x]
            self.X.append(x)
            self.y.append(float(v))
            for i, p in enumerate(self.pending):
                if np.allclose(p, x):
                    self.pending.pop(i)
                    break

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path) as f:
            o.__dict__.update(json.load(f))
        return o
