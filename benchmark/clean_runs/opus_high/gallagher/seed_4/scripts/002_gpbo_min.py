import json
import numpy as np
import torch
from scipy.stats import qmc


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_init=10):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.dim = len(self.bounds)
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.X, self.y = [], []
        self.pending = []
        self.n_asked = 0
        self.rng_state = np.random.default_rng(self.seed).bit_generator.state

    def _rng(self):
        r = np.random.default_rng()
        r.bit_generator.state = self.rng_state
        return r

    def _unit(self, X):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(X, float) - lo) / (hi - lo)

    def _from_unit(self, U):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return np.clip(lo + np.asarray(U) * (hi - lo), lo, hi)

    def _init_point(self, i):
        sob = qmc.Sobol(self.dim, scramble=True, seed=self.seed).random(max(16, self.n_init))
        return sob[i % len(sob)]

    def _bo_point(self, rng):
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        U = torch.tensor(self._unit(self.X), dtype=torch.double)
        y = np.asarray(self.y, float)
        Y = torch.tensor(y, dtype=torch.double).unsqueeze(-1)
        Ys = (Y - Y.mean()) / (Y.std() if Y.std() > 0 else 1.0)
        torch.manual_seed(int(rng.integers(2**31)))
        gp = SingleTaskGP(U, Ys)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        acq = LogExpectedImprovement(gp, best_f=Ys.min(), maximize=False)
        b = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.double)
        cand, _ = optimize_acqf(acq, bounds=b, q=1, num_restarts=10, raw_samples=512)
        u = cand.detach().numpy()[0]
        # avoid exact duplicates
        dmin = np.min(np.linalg.norm(self._unit(self.X) - u, axis=1))
        if dmin < 1e-6:
            best = self._unit(self.X)[int(np.argmin(y))]
            u = np.clip(best + rng.normal(0, 0.01, self.dim), 0, 1)
        return u

    def ask(self, n):
        rng = self._rng()
        out = []
        for _ in range(n):
            if len(self.X) < self.n_init or len(self.X) < 2:
                u = self._init_point(self.n_asked)
            else:
                u = self._bo_point(rng)
            self.n_asked += 1
            x = self._from_unit(u).tolist()
            self.pending.append(x)
            out.append(x)
        self.rng_state = rng.bit_generator.state
        return out

    def tell(self, X, y, g):
        for xi, yi in zip(X, y):
            self.X.append([float(v) for v in xi])
            self.y.append(float(yi) if np.isfinite(yi) else float("nan"))
            if list(xi) in self.pending:
                self.pending.remove(list(xi))
        yy = np.array(self.y)
        if np.isnan(yy).any():
            fin = yy[~np.isnan(yy)]
            worst = fin.max() + 1.0 if len(fin) else 1.0
            self.y = [worst if np.isnan(v) else v for v in yy.tolist()]

    def save(self, path):
        d = {k: getattr(self, k) for k in
             ["seed", "n_init", "X", "y", "pending", "n_asked"]}
        d["bounds"] = self.bounds.tolist()
        d["rng_state"] = self.rng_state
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls.__new__(cls)
        o.bounds = np.array(d.pop("bounds"))
        o.dim = len(o.bounds)
        for k, v in d.items():
            setattr(o, k, v)
        return o
