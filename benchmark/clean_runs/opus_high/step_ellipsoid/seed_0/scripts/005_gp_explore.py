import pickle

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc


class Optimizer:
    """Global GP search outside a known basin.

    A GP on log(y - floor) over the whole box; each point minimizes the lower bound
    mean - beta * std over Sobol and local candidates. Candidates within `excl_r`
    (user units) of any point from the first tell (the warm start) with y < `excl_y`
    are excluded, so the search looks for other basins.
    """

    def __init__(self, space, seed=0, beta=2.0, excl_y=100.0, excl_r=1.5,
                 n_sobol=4096, n_local=1500, local_sd=0.4, max_fit=1200):
        self.lb = np.array(space["bounds"], dtype=float)[:, 0]
        self.ub = np.array(space["bounds"], dtype=float)[:, 1]
        self.d = len(self.lb)
        self.seed = int(seed)
        self.beta = float(beta)
        self.excl_y, self.excl_r = float(excl_y), float(excl_r)
        self.n_sobol, self.n_local = int(n_sobol), int(n_local)
        self.local_sd = float(local_sd)
        self.max_fit = int(max_fit)
        self.rng = np.random.default_rng(self.seed)
        self.X, self.y = [], []
        self.centres = None  # exclusion centres, set at the first tell
        self.pending = []

    def _excluded(self, C):
        if self.centres is None or len(self.centres) == 0:
            return np.zeros(len(C), bool)
        dmin = np.full(len(C), np.inf)
        for i in range(0, len(self.centres), 200):
            c = self.centres[i:i + 200]
            dmin = np.minimum(dmin, np.linalg.norm(C[:, None, :] - c[None], axis=2).min(1))
        return dmin < self.excl_r

    def ask(self, n):
        X = np.array(self.X + self.pending, float)
        y = np.array(self.y, float)
        out = []
        for _ in range(n):
            out.append(self._next(np.array(self.X, float), y, X))
            X = np.vstack([X, out[-1]]) if len(X) else np.array([out[-1]])
        self.pending += [list(p) for p in out]
        return [list(map(float, p)) for p in out]

    def _next(self, Xt, yt, Xall):
        if len(yt) < 5:
            return self.rng.uniform(self.lb, self.ub)
        ok = np.isfinite(yt)
        Xt, yt = Xt[ok], yt[ok]
        if len(yt) > self.max_fit:  # keep the best half, subsample the rest
            o = np.argsort(yt)
            keep = np.r_[o[:self.max_fit // 2],
                         self.rng.choice(o[self.max_fit // 2:], self.max_fit // 2, replace=False)]
            Xt, yt = Xt[keep], yt[keep]
        z = np.log(yt - yt.min() + 1.0)
        tX = torch.tensor(Xt, dtype=torch.double)
        tz = torch.tensor(z, dtype=torch.double).unsqueeze(-1)
        bounds = torch.tensor(np.c_[self.lb, self.ub].T, dtype=torch.double)
        gp = SingleTaskGP(tX, tz, input_transform=Normalize(self.d, bounds=bounds),
                          outcome_transform=Standardize(1))
        torch.manual_seed(int(self.rng.integers(1, 2**31)))
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        except Exception:
            pass
        sob = qmc.Sobol(self.d, scramble=True, seed=int(self.rng.integers(1, 2**31)))
        C = self.lb + sob.random(self.n_sobol) * (self.ub - self.lb)
        # local candidates around the best points outside the excluded zone
        Xo = np.array(self.X, float)
        yo = np.array(self.y, float)
        free = ~self._excluded(Xo)
        if free.any():
            idx = np.where(free)[0][np.argsort(yo[free])[:10]]
            base = Xo[self.rng.choice(idx, self.n_local)]
            sd = self.local_sd * self.rng.choice([0.25, 0.5, 1.0, 2.0], (self.n_local, 1))
            L = np.clip(base + sd * self.rng.standard_normal(base.shape), self.lb, self.ub)
            C = np.vstack([C, L])
        C = C[~self._excluded(C)]
        if len(Xall):  # avoid near-repeats
            far = np.ones(len(C), bool)
            for i in range(0, len(Xall), 300):
                far &= np.linalg.norm(C[:, None] - Xall[None, i:i + 300], axis=2).min(1) > 0.05
            C = C[far]
        if len(C) == 0:
            return self.rng.uniform(self.lb, self.ub)
        with torch.no_grad():
            post = gp.posterior(torch.tensor(C, dtype=torch.double))
            m = post.mean.squeeze(-1).numpy()
            s = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
        return C[np.argmin(m - self.beta * s)]

    def tell(self, X, y, g):
        X = [list(map(float, x)) for x in X]
        y = [float(v) if np.isfinite(v) else np.nan for v in y]
        if self.centres is None:
            a, b = np.array(X, float), np.array(y, float)
            self.centres = a[np.nan_to_num(b, nan=np.inf) < self.excl_y] if len(a) else np.zeros((0, self.d))
        fin = [v for v in self.y + y if np.isfinite(v)]
        worst = max(fin) if fin else 1e6
        for x, v in zip(X, y):
            self.X.append(x)
            self.y.append(v if np.isfinite(v) else worst)
            for i, p in enumerate(self.pending):
                if np.allclose(p, x, atol=1e-9):
                    self.pending.pop(i)
                    break

    def save(self, path):
        d = dict(self.__dict__)
        d["rng"] = self.rng.bit_generator.state
        with open(path, "wb") as f:
            pickle.dump(d, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        o = cls.__new__(cls)
        rs = d.pop("rng")
        o.__dict__.update(d)
        o.rng = np.random.default_rng()
        o.rng.bit_generator.state = rs
        return o
