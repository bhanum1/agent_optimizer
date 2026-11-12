import json
import numpy as np
import torch


class Optimizer:
    """GP-BO (botorch SingleTaskGP + LogEI) on the unit-normalized box.

    Every `explore_every`-th proposal maximizes UCB with a large beta instead,
    to keep probing a possibly multimodal landscape.
    """

    def __init__(self, space, seed=0, explore_every=4, beta=9.0, n_init=6):
        self.bounds = [list(b) for b in space["bounds"]]
        self.seed = int(seed)
        self.explore_every = int(explore_every)
        self.beta = float(beta)
        self.n_init = int(n_init)
        self.X, self.y = [], []
        self.pending = []
        self.n_asked = 0

    def _lo_hi(self):
        b = np.array(self.bounds, dtype=float)
        return b[:, 0], b[:, 1]

    def _random(self, rng):
        lo, hi = self._lo_hi()
        return list(lo + rng.random(len(lo)) * (hi - lo))

    def _propose(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        rng = np.random.default_rng(self.seed * 1000 + self.n_asked)
        if len(self.y) < self.n_init:
            return self._random(rng)
        torch.manual_seed(self.seed * 1000 + self.n_asked)
        lo, hi = self._lo_hi()
        Xn = (np.array(self.X) - lo) / (hi - lo)
        y = np.array(self.y, dtype=float)
        ys = (y - y.mean()) / (y.std() + 1e-12)
        tX = torch.tensor(Xn, dtype=torch.double)
        tY = -torch.tensor(ys, dtype=torch.double).unsqueeze(-1)  # botorch maximizes
        model = SingleTaskGP(tX, tY)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        if self.explore_every > 0 and self.n_asked % self.explore_every == self.explore_every - 1:
            acq = UpperConfidenceBound(model, beta=self.beta)
        else:
            acq = LogExpectedImprovement(model, best_f=tY.max())
        ub = torch.stack([torch.zeros(len(lo)), torch.ones(len(lo))]).double()
        cand, _ = optimize_acqf(acq, bounds=ub, q=1, num_restarts=10, raw_samples=1024)
        x = cand.detach().numpy()[0]
        # avoid re-measuring a known point on a deterministic problem
        if len(Xn) and np.min(np.linalg.norm(Xn - x, axis=1)) < 1e-4:
            x = rng.random(len(lo))
        return list(lo + np.clip(x, 0, 1) * (hi - lo))

    def ask(self, n):
        out = []
        for _ in range(n):
            x = [float(v) for v in self._propose()]
            self.n_asked += 1
            self.pending.append(x)
            out.append(x)
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            xi = [float(v) for v in xi]
            self.X.append(xi)
            self.y.append(float(yi))
            if xi in self.pending:
                self.pending.remove(xi)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path) as f:
            obj.__dict__.update(json.load(f))
        return obj
