"""TuRBO-1 style trust-region BO (sequential), with GP-EI global restarts,
set up for exploring away from known basins: starts in global mode with the given
`avoid` centers (user units) excluded within `excl` (unit-cube distance), and fits
the global GP on a thinned subset so dense local clusters don't swamp it.

State is plain data (X, y, TR length, counters, rng) so save/load is exact.
"""
import json
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm, qmc

torch.set_default_dtype(torch.float64)


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, length_init=0.8, length_min=0.5 ** 7,
                 length_max=1.6, succ_tol=3, fail_tol=5, n_cand=4000,
                 max_fit=300, log_y=False, avoid=None, excl=0.2, thin=0.03):
        self.lb = [b[0] for b in space["bounds"]]
        self.ub = [b[1] for b in space["bounds"]]
        self.d = len(self.lb)
        self.cfg = dict(seed=seed, length_init=length_init, length_min=length_min,
                        length_max=length_max, succ_tol=succ_tol, fail_tol=fail_tol,
                        n_cand=n_cand, max_fit=max_fit, log_y=log_y,
                        excl=excl, thin=thin)
        self.X, self.y = [], []
        self.length = length_init
        self.succ = self.fail = 0
        self.center = None          # index into X of the TR center
        self.best_tr = None         # best y within current TR run
        self.restarts = (self._unit(avoid).tolist() if avoid else [])  # unit coords
        self.pending = []
        self.mode = "global" if avoid else "tr"  # "tr" or "global" (next ask is a global EI point)
        self.rng = np.random.default_rng(seed)

    # ---- helpers
    def _unit(self, X):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return (np.asarray(X, float) - lb) / (ub - lb)

    def _user(self, U):
        lb, ub = np.array(self.lb), np.array(self.ub)
        return lb + np.clip(U, 0, 1) * (ub - lb)

    def _thin(self, U, y):
        # greedy: visit points best-first, keep one if no kept point within `thin`
        order = np.argsort(y)
        keep = []
        for i in order:
            if not keep or np.min(np.linalg.norm(U[keep] - U[i], axis=1)) > self.cfg["thin"]:
                keep.append(i)
        keep = np.array(keep)
        if len(keep) > self.cfg["max_fit"]:
            keep = np.concatenate([keep[:50], self.rng.choice(keep[50:], self.cfg["max_fit"] - 50,
                                                               replace=False)])
        return U[keep], y[keep]

    def _fit(self, U, y):
        Y = np.asarray(y, float)
        if self.cfg["log_y"]:
            Y = np.log(Y - Y.min() + 1e-3 * (np.ptp(Y) + 1e-12))
        mu, sd = Y.mean(), Y.std() + 1e-12
        Yt = torch.tensor(((Y - mu) / sd)[:, None])
        Ut = torch.tensor(U)
        gp = SingleTaskGP(Ut, Yt)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        gp.eval()
        return gp, (Y - mu) / sd

    def _post(self, gp, C):
        with torch.no_grad():
            p = gp.posterior(torch.tensor(C))
            return p.mean.numpy().ravel(), p.variance.clamp_min(1e-12).sqrt().numpy().ravel()

    # ---- ask/tell
    def ask(self, n):
        out = []
        for _ in range(n):
            out.append(self._ask_one())
        self.pending.extend(out)
        return out

    def _ask_one(self):
        U = self._unit(self.X)
        y = np.array(self.y, float)
        nc = self.cfg["n_cand"]
        if len(y) < 2 * self.d:
            return self._user(self.rng.random(self.d)).tolist()
        if self.mode == "global":
            gp, Yt = self._fit(*self._thin(U, y))
            C = self.rng.random((nc * 2, self.d))
            # stay away from converged centers
            if self.restarts:
                R = np.array(self.restarts)
                dmin = np.min(np.linalg.norm(C[:, None, :] - R[None], axis=2), axis=1)
                ex = self.cfg["excl"]
                C = C[dmin > ex] if np.sum(dmin > ex) > 10 else C
            m, s = self._post(gp, C)
            best = Yt.min()
            z = (best - m) / s
            ei = s * (z * norm.cdf(z) + norm.pdf(z))
            return self._user(C[np.argmax(ei)]).tolist()
        # trust region around center
        c = U[self.center]
        # local data for GP
        if len(y) > self.cfg["max_fit"]:
            idx = np.argsort(np.linalg.norm(U - c, axis=1))[: self.cfg["max_fit"]]
        else:
            idx = np.arange(len(y))
        gp, Yt = self._fit(U[idx], y[idx])
        try:
            ls = gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        except AttributeError:
            ls = gp.covar_module.lengthscale.detach().numpy().ravel()
        w = ls / ls.mean()
        w = w / np.prod(w) ** (1.0 / self.d)
        lo = np.clip(c - w * self.length / 2, 0, 1)
        hi = np.clip(c + w * self.length / 2, 0, 1)
        sob = qmc.Sobol(self.d, scramble=True, seed=self.rng.integers(2**31)).random(nc)
        pert = lo + (hi - lo) * sob
        prob = min(20.0 / self.d, 1.0)
        mask = self.rng.random((nc, self.d)) <= prob
        mask[np.arange(nc), self.rng.integers(0, self.d, nc)] = True
        C = np.where(mask, pert, c)
        # Thompson sample
        with torch.no_grad():
            post = gp.posterior(torch.tensor(C))
            f = post.rsample(torch.Size([1])).numpy().ravel()
        return self._user(C[np.argmin(f)]).tolist()

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = [float(t) for t in x]
            v = float(v) if np.isfinite(v) else (max(self.y) if self.y else 1e10)
            self.X.append(x)
            self.y.append(v)
            asked = None
            for i, p in enumerate(self.pending):
                if np.allclose(p, x):
                    asked = i
                    break
            if asked is None:
                continue  # warm-start / foreign point: just data
            self.pending.pop(asked)
            self._update(len(self.y) - 1)
        if self.center is None and self.y:
            self.center = int(np.argmin(self.y))
            self.best_tr = self.y[self.center]

    def _update(self, i):
        v = self.y[i]
        if self.mode == "global":
            self.mode = "tr"
            self.length = self.cfg["length_init"]
            self.succ = self.fail = 0
            # start new TR at the global point unless it's worse than... just use it
            self.center = i
            self.best_tr = v
            return
        if self.center is None:
            self.center = int(np.argmin(self.y))
            self.best_tr = self.y[self.center]
            return
        if v < self.best_tr - 1e-3 * abs(self.best_tr):
            self.succ += 1
            self.fail = 0
        else:
            self.succ = 0
            self.fail += 1
        if v < self.best_tr:
            self.best_tr = v
            self.center = i
        if self.succ >= self.cfg["succ_tol"]:
            self.length = min(2 * self.length, self.cfg["length_max"])
            self.succ = 0
        elif self.fail >= self.cfg["fail_tol"]:
            self.length /= 2
            self.fail = 0
        if self.length < self.cfg["length_min"]:
            self.restarts.append(self._unit([self.X[self.center]])[0].tolist())
            self.mode = "global"

    def save(self, path):
        st = dict(lb=self.lb, ub=self.ub, d=self.d, cfg=self.cfg, X=self.X, y=self.y,
                  length=self.length, succ=self.succ, fail=self.fail, center=self.center,
                  best_tr=self.best_tr, restarts=self.restarts, pending=self.pending,
                  mode=self.mode, rng=self.rng.bit_generator.state)
        with open(path, "w") as f:
            json.dump(st, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            st = json.load(f)
        o = cls.__new__(cls)
        rng = st.pop("rng")
        o.__dict__.update(st)
        o.rng = np.random.default_rng()
        o.rng.bit_generator.state = rng
        return o
