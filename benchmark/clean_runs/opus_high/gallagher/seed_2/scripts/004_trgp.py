import json
import numpy as np
import torch


class Optimizer:
    """Trust-region GP-BO around the incumbent.

    A GP is fitted to the `k_near` points nearest the best point; LogEI is
    maximized inside a box of half-width `radius` (user units) around it.
    The radius grows by `grow` after an improvement and halves after
    `fail_tol` consecutive non-improving points.
    """

    def __init__(self, space, seed=0, radius=0.4, k_near=12, grow=1.5,
                 fail_tol=2, min_radius=0.03, max_radius=1.0):
        self.bounds = [list(b) for b in space["bounds"]]
        self.seed = int(seed)
        self.radius = float(radius)
        self.k_near = int(k_near)
        self.grow = float(grow)
        self.fail_tol = int(fail_tol)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.X, self.y = [], []
        self.pending = []
        self.n_asked = 0
        self.fails = 0
        self.n_told = 0

    def _propose(self):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        rng = np.random.default_rng(self.seed * 1000 + self.n_asked)
        torch.manual_seed(self.seed * 1000 + self.n_asked)
        b = np.array(self.bounds, dtype=float)
        lo, hi = b[:, 0], b[:, 1]
        X = np.array(self.X, dtype=float)
        y = np.array(self.y, dtype=float)
        c = X[np.argmin(y)]
        tlo = np.maximum(lo, c - self.radius)
        thi = np.minimum(hi, c + self.radius)
        idx = np.argsort(np.linalg.norm(X - c, axis=1))[: self.k_near]
        Xl, yl = X[idx], y[idx]
        scale = 3 * self.radius
        Xn = (Xl - c) / scale + 0.5
        ys = (yl - yl.mean()) / (yl.std() + 1e-12)
        tX = torch.tensor(Xn, dtype=torch.double)
        tY = -torch.tensor(ys, dtype=torch.double).unsqueeze(-1)
        model = SingleTaskGP(tX, tY)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        except Exception:
            pass
        acq = LogExpectedImprovement(model, best_f=tY.max())
        ub = torch.tensor(np.stack([(tlo - c) / scale + 0.5,
                                    (thi - c) / scale + 0.5]), dtype=torch.double)
        cand, _ = optimize_acqf(acq, bounds=ub, q=1, num_restarts=10, raw_samples=512)
        x = (cand.detach().numpy()[0] - 0.5) * scale + c
        if np.min(np.linalg.norm(X - x, axis=1)) < 1e-4:
            x = tlo + rng.random(len(lo)) * (thi - tlo)
        return list(np.clip(x, lo, hi))

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
            own = xi in self.pending
            best = min(self.y) if self.y else float("inf")
            self.X.append(xi)
            self.y.append(float(yi))
            if own:
                self.pending.remove(xi)
                # trust-region update only for points this optimizer proposed
                if yi < best - 1e-9:
                    self.fails = 0
                    self.radius = min(self.max_radius, self.radius * self.grow)
                else:
                    self.fails += 1
                    if self.fails >= self.fail_tol:
                        self.fails = 0
                        self.radius = max(self.min_radius, self.radius / 2)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path) as f:
            obj.__dict__.update(json.load(f))
        return obj
