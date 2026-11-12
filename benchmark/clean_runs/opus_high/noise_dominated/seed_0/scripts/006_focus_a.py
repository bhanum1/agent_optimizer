"""Single focused local BO with an outlier-trimmed GP, for the plateau endgame.

One box, recentred each ask on the observed point with the lowest posterior
mean (among the `n_local` points nearest the centre). Box half-width per
dimension is `k_ls` times the GP lengthscale (unit box), clipped to
[`half_min`, `half_max`]. Inside it, LogEI against the best posterior mean.
Every `rep_every`-th ask (0 = never) instead re-measures the current
lowest-posterior-mean point, since the score is the lowest single draw.
No killing, no taboo, no spawning. `center` is in user units.
"""
import json
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")


class Optimizer:
    max_ask = 1

    def __init__(self, space, center, seed=0, n_local=200, k_ls=1.0,
                 half_min=0.005, half_max=0.08, rep_every=0):
        self.lb = [float(b[0]) for b in space["bounds"]]
        self.ub = [float(b[1]) for b in space["bounds"]]
        self.d = len(self.lb)
        self.seed = int(seed)
        self.center = self._to_unit(center).tolist()
        self.n_local = int(n_local)
        self.k_ls = float(k_ls)
        self.half_min = float(half_min)
        self.half_max = float(half_max)
        self.rep_every = int(rep_every)
        self.X, self.y = [], []
        self.n_ask = 0
        self.rng_state = np.random.default_rng(self.seed).bit_generator.state

    def _rng(self):
        r = np.random.default_rng()
        r.bit_generator.state = self.rng_state
        return r

    def _to_unit(self, X):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return (np.asarray(X, float) - lb) / (ub - lb)

    def _from_unit(self, U):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return lb + np.clip(U, 0, 1) * (ub - lb)

    def _fit1(self, X, y):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        torch.manual_seed(self.seed + self.n_ask)
        y = np.asarray(y, float)
        spread = max(np.std(y), 1e-9)
        z = np.log(y - y.min() + 0.1 * spread)
        z = (z - z.mean()) / max(z.std(), 1e-9)
        gp = SingleTaskGP(torch.tensor(X, dtype=torch.double),
                          torch.tensor(z, dtype=torch.double).unsqueeze(-1))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        return gp, z

    def _fit(self, X, y):
        import torch
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        gp, z = self._fit1(X, y)
        if len(y) < 20:
            return gp
        with torch.no_grad():
            mu = gp.posterior(torch.tensor(X, dtype=torch.double)).mean.squeeze(-1).numpy()
        r = z - mu
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        keep = r < np.median(r) + 3 * s
        if keep.all() or keep.sum() < 10:
            return gp
        return self._fit1(X[keep], y[keep])[0]

    @staticmethod
    def _ls(gp):
        try:
            ls = gp.covar_module.base_kernel.lengthscale
        except AttributeError:
            ls = gp.covar_module.lengthscale
        return ls.detach().numpy().ravel()

    def ask(self, n):
        return [self._ask_one() for _ in range(n)]

    def _ask_one(self):
        import torch
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        rng = self._rng()
        self.n_ask += 1
        c = np.asarray(self.center)
        X = self._to_unit(self.X) if self.X else np.zeros((0, self.d))
        y = np.asarray(self.y, float)
        if len(y) < 10:
            u = np.clip(c + 2 * self.half_min * (rng.random(self.d) - 0.5), 0, 1)
        else:
            idx = np.argsort(np.linalg.norm(X - c, axis=1))[: self.n_local]
            gp = self._fit(X[idx], y[idx])
            half = np.clip(self.k_ls * self._ls(gp), self.half_min, self.half_max)
            with torch.no_grad():
                mu = gp.posterior(torch.tensor(X[idx], dtype=torch.double)).mean.squeeze(-1).numpy()
            inside = (np.abs(X[idx] - c) <= half).all(axis=1)
            if inside.any():
                k = np.where(inside)[0][np.argmin(mu[inside])]
            else:
                k = int(np.argmin(mu))
            c = X[idx][k]
            self.center = c.tolist()
            if self.rep_every and self.n_ask % self.rep_every == 0:
                u = c
            else:
                lo = np.clip(c - half, 0, 1)
                hi = np.clip(c + half, 0, 1)
                acq = LogExpectedImprovement(gp, best_f=float(mu[k]), maximize=False)
                torch.manual_seed(int(rng.integers(1 << 30)))
                try:
                    cand, _ = optimize_acqf(
                        acq, bounds=torch.tensor(np.vstack([lo, hi]), dtype=torch.double),
                        q=1, num_restarts=8, raw_samples=256)
                    u = cand.detach().numpy().ravel()
                except Exception:
                    u = lo + rng.random(self.d) * (hi - lo)
        x = self._from_unit(u).tolist()
        self.rng_state = rng.bit_generator.state
        return x

    def tell(self, X, y, g=None):
        for xi, yi in zip(X, y):
            yi = float(yi)
            if not math.isfinite(yi):
                yi = max(self.y) if self.y else 1e9
            self.X.append([float(v) for v in xi])
            self.y.append(yi)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        o = cls.__new__(cls)
        o.__dict__.update(d)
        return o
