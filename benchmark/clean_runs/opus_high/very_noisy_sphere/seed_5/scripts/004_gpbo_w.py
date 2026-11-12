import json
import numpy as np
import torch


class Optimizer:
    """GP-BO: SingleTaskGP (learned noise) + qLogNoisyExpectedImprovement.

    Optional trust region: if tr_frac < 1, candidates are restricted to a box of
    side tr_frac*range centered on the best posterior-mean observed point.
    If warp_c > 0, the GP is fit to log(y - min(y) + warp_c), which tames the
    heavy one-sided (upward) noise and stretches the region near the floor.
    """
    max_ask = 1

    def __init__(self, space, seed=0, n_init=0, tr_frac=1.0, raw_samples=512,
                 num_restarts=8, max_fit=400, warp_c=0.0):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lo)
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.tr_frac = float(tr_frac)
        self.raw_samples = int(raw_samples)
        self.num_restarts = int(num_restarts)
        self.max_fit = int(max_fit)
        self.warp_c = float(warp_c)
        self.X, self.y = [], []
        self.pending = []
        self.n_asked = 0

    def _rng(self):
        return np.random.default_rng(self.seed * 100003 + self.n_asked)

    def ask(self, n):
        out = []
        for _ in range(n):
            x = self._one()
            self.n_asked += 1
            self.pending.append(x)
            out.append(x)
        return out

    def _one(self):
        rng = self._rng()
        if len(self.y) < max(self.n_init, 5):
            return list(self.lo + rng.random(self.d) * (self.hi - self.lo))
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.optim import optimize_acqf
        torch.manual_seed(self.seed * 7919 + self.n_asked)
        X = np.array(self.X, float)
        y = np.array(self.y, float)
        # subset: keep best points + random others if too many
        if len(y) > self.max_fit:
            o = np.argsort(y)
            keep = list(o[: self.max_fit // 2])
            rest = o[self.max_fit // 2:]
            keep += list(rng.choice(rest, self.max_fit - len(keep), replace=False))
            X, y = X[keep], y[keep]
        if self.warp_c > 0:
            y = np.log(y - y.min() + self.warp_c)
        Xu = (X - self.lo) / (self.hi - self.lo)
        tX = torch.tensor(Xu, dtype=torch.double)
        ty = torch.tensor(-y, dtype=torch.double).unsqueeze(-1)  # botorch maximizes
        model = SingleTaskGP(tX, ty, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        bounds = torch.stack([torch.zeros(self.d), torch.ones(self.d)]).double()
        if self.tr_frac < 1.0:
            with torch.no_grad():
                mu = model.posterior(tX).mean.squeeze(-1)
            c = Xu[int(torch.argmax(mu))]
            h = self.tr_frac / 2
            bounds = torch.tensor(np.stack([np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)]))
        acq = qLogNoisyExpectedImprovement(model, X_baseline=tX, prune_baseline=True)
        try:
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=self.num_restarts,
                                    raw_samples=self.raw_samples)
            xu = cand.detach().numpy()[0]
        except Exception:
            xu = rng.random(self.d)
        x = self.lo + np.clip(xu, 0, 1) * (self.hi - self.lo)
        return [float(v) for v in x]

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            xi = [float(v) for v in xi]
            self.X.append(xi)
            self.y.append(float(yi))
            for j, p in enumerate(self.pending):
                if np.allclose(p, xi):
                    self.pending.pop(j)
                    break

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__ | {"lo": self.lo.tolist(), "hi": self.hi.tolist()}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            s = json.load(f)
        o = cls.__new__(cls)
        o.__dict__.update(s)
        o.lo = np.array(s["lo"])
        o.hi = np.array(s["hi"])
        return o
