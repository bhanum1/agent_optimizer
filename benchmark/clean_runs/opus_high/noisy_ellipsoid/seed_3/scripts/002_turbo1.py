import json, pickle
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)


class Optimizer:
    """TuRBO-1 style: GP on log-transformed y (local subset), LogEI inside a trust region
    centred on the best posterior-mean point."""
    max_ask = 1

    def __init__(self, space, seed=0, L0=0.8, Lmin=0.01, Lmax=1.6, succ_tol=3, fail_tol=6,
                 n_local=300):
        self.lo = np.array(space["bounds"])[:, 0] if "bounds" in space else None
        b = np.array(space["bounds"], dtype=float)
        self.lo, self.hi = b[:, 0], b[:, 1]
        self.d = len(self.lo)
        self.L, self.L0, self.Lmin, self.Lmax = L0, L0, Lmin, Lmax
        self.succ_tol, self.fail_tol, self.n_local = succ_tol, fail_tol, n_local
        self.succ = self.fail = 0
        self.X, self.y = [], []
        self.best_seen = np.inf
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.n_restarts_tr = 0

    def _u(self, X):
        return (np.asarray(X) - self.lo) / (self.hi - self.lo)

    def _ty(self, y):
        y = np.asarray(y, float)
        q = np.percentile(y, [25, 75])
        c = max(1e-3 * (q[1] - q[0]), 1e-12)
        return np.log(y - y.min() + c)

    def _fit(self, U, t):
        Ut = torch.tensor(U); tt = torch.tensor(t).unsqueeze(-1)
        mu, sd = tt.mean(), tt.std().clamp_min(1e-9)
        m = SingleTaskGP(Ut, (tt - mu) / sd)
        try:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
        except Exception:
            pass
        return m

    def ask(self, n):
        out = []
        for _ in range(n):
            out.append(self._ask1())
        return out

    def _ask1(self):
        if len(self.y) < 5:
            return list(self.lo + self.rng.random(self.d) * (self.hi - self.lo))
        U = self._u(self.X); y = np.array(self.y)
        t = self._ty(y)
        # centre: best observed among top candidates by GP mean
        ib = int(np.argmin(y))
        dist = np.linalg.norm(U - U[ib], axis=1)
        idx = np.argsort(dist)[: self.n_local]
        m = self._fit(U[idx], t[idx])
        with torch.no_grad():
            mu = m.posterior(torch.tensor(U[idx])).mean.squeeze(-1).numpy()
        ic = idx[int(np.argmin(mu))]
        centre = U[ic]
        ls = m.covar_module.lengthscale.detach().numpy().ravel() if hasattr(m.covar_module, "lengthscale") \
            else m.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        w = ls / np.exp(np.mean(np.log(ls)))
        tr_lo = np.clip(centre - self.L * w / 2, 0, 1)
        tr_hi = np.clip(centre + self.L * w / 2, 0, 1)
        acq = LogExpectedImprovement(m, best_f=float(mu.min()), maximize=False)
        torch.manual_seed(int(self.rng.integers(1 << 30)))
        try:
            cand, _ = optimize_acqf(acq, bounds=torch.tensor(np.vstack([tr_lo, tr_hi])), q=1,
                                    num_restarts=10, raw_samples=512)
            u = cand.detach().numpy().ravel()
        except Exception:
            u = tr_lo + self.rng.random(self.d) * (tr_hi - tr_lo)
        return list(self.lo + np.clip(u, 0, 1) * (self.hi - self.lo))

    def tell(self, X, y, g):
        y = [float(v) for v in y]
        if len(y) == 1 and len(self.y) >= 5:
            if y[0] < self.best_seen - 1e-3 * abs(self.best_seen):
                self.succ += 1; self.fail = 0
            else:
                self.fail += 1; self.succ = 0
            if self.succ >= self.succ_tol:
                self.L = min(2 * self.L, self.Lmax); self.succ = 0
            if self.fail >= self.fail_tol:
                self.L /= 2; self.fail = 0
            if self.L < self.Lmin:
                self.L = self.L0; self.n_restarts_tr += 1
        self.X += [list(map(float, x)) for x in X]
        self.y += y
        self.best_seen = min(self.best_seen, min(y))

    def save(self, path):
        st = dict(self.__dict__); st["rng"] = self.rng.bit_generator.state
        with open(path, "wb") as f:
            pickle.dump(st, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            st = pickle.load(f)
        o = cls.__new__(cls)
        rs = st.pop("rng"); o.__dict__.update(st)
        o.rng = np.random.default_rng(); o.rng.bit_generator.state = rs
        return o
