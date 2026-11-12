"""TuRBO-1 style trust-region BO for a noisy continuous box.

Sobol initial design, then a Matern-5/2 ARD GP (fit on the points nearest the
trust-region centre) with Thompson sampling over perturbed candidates inside the
trust region. The centre is the observed point with the lowest GP posterior mean,
which is more robust to noise than the lowest raw value. The region grows on
success streaks, shrinks on failure streaks, and restarts with a fresh Sobol design
when it collapses (all data is kept for modelling).

turbo2: the GP is fit on z = log(y - min(y) + shift), clipped at an upper quantile,
because the noise is heavy-tailed upward (replicates of a ~400 point gave up to 1050).
A warm-start tell (points not asked by this optimizer) is taken as the initial design.
_propose is guarded: any numerical error falls back to a random point in the TR.
"""
import json
import warnings

import numpy as np
import torch
from scipy.stats import qmc

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore")


class Optimizer:
    max_ask = 1
    batch = False

    def __init__(self, space, seed=0, n_init=32, n_cand=2048, max_train=300,
                 length_init=0.8, length_min=0.5 ** 7, length_max=1.6,
                 succ_tol=3, fail_tol=None, keep_data_on_restart=True,
                 n_init_restart=12, shift=10.0, clip_q=0.85):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.dim = len(self.lo)
        self.seed = int(seed)
        self.n_init = int(n_init)
        self.n_cand = int(n_cand)
        self.max_train = int(max_train)
        self.length_init = float(length_init)
        self.length_min = float(length_min)
        self.length_max = float(length_max)
        self.succ_tol = int(succ_tol)
        self.fail_tol = int(fail_tol) if fail_tol else max(4, self.dim)
        self.keep_data = bool(keep_data_on_restart)
        self.n_init_restart = int(n_init_restart)
        self.shift = float(shift)
        self.clip_q = float(clip_q)
        self.rng = np.random.default_rng(self.seed)
        self.X = []          # unit-cube points
        self.y = []
        self.pending = []    # unit-cube points asked, not yet told
        self.init_queue = []
        self.restart_from = 0  # index into X where the current TR run started
        self.n_restarts = 0
        self._new_run()

    # ---- trust-region bookkeeping ----
    def _new_run(self):
        self.length = self.length_init
        self.succ = 0
        self.fail = 0
        self.restart_from = len(self.X)
        sob = qmc.Sobol(self.dim, scramble=True, seed=self.rng.integers(2 ** 31))
        n = self.n_init if not self.X else self.n_init_restart
        self.init_queue = sob.random(n).tolist()
        self.run_init = n
        self.best_run = None

    def _to_unit(self, x):
        return (np.asarray(x, float) - self.lo) / (self.hi - self.lo)

    def _from_unit(self, u):
        return (self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)).tolist()

    # ---- GP ----
    def _transform(self, yt):
        z = np.log(np.asarray(yt, float) - np.min(self.y) + self.shift)
        return np.minimum(z, np.quantile(z, self.clip_q))

    def _fit(self, Xt, yt):
        yt = self._transform(yt)
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.models.transforms import Standardize
        Xt = torch.tensor(Xt)
        yt = torch.tensor(yt).unsqueeze(-1)
        model = SingleTaskGP(Xt, yt, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        torch.manual_seed(int(self.rng.integers(2 ** 31)))
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
        return model

    def _train_set(self, center):
        X = np.array(self.X)
        y = np.array(self.y)
        if len(X) <= self.max_train:
            return X, y
        d = np.linalg.norm(X - center, axis=1)
        idx = np.argsort(d)[: self.max_train]
        return X[idx], y[idx]

    def _center(self, model, X):
        with torch.no_grad():
            mu = model.posterior(torch.tensor(X)).mean.squeeze(-1).numpy()
        return X[int(np.argmin(mu))]

    def _propose(self):
        try:
            return self._propose_gp()
        except Exception:
            yr = np.array(self.y[self.restart_from:])
            c = np.array(self.X[self.restart_from + int(np.argmin(yr))])
            h = self.length / 2
            return np.clip(c + self.rng.uniform(-h, h, self.dim), 0, 1)

    def _propose_gp(self):
        Xall = np.array(self.X)
        run = Xall[self.restart_from:] if not self.keep_data else Xall
        yall = np.array(self.y)
        yrun = yall[self.restart_from:] if not self.keep_data else yall
        # first fit around the raw best of the current run, then pick the centre by posterior mean
        yr = yall[self.restart_from:]
        c0 = Xall[self.restart_from + int(np.argmin(yr))]
        Xt, yt = self._train_set(c0) if self.keep_data else (run, yrun)
        model = self._fit(Xt, yt)
        Xrun = Xall[self.restart_from:]
        center = self._center(model, Xrun)
        try:
            ls = model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        except AttributeError:
            ls = model.covar_module.lengthscale.detach().numpy().ravel()
        w = ls / ls.mean()
        w = w / np.prod(w) ** (1.0 / self.dim)
        lb = np.clip(center - w * self.length / 2, 0, 1)
        ub = np.clip(center + w * self.length / 2, 0, 1)
        sob = qmc.Sobol(self.dim, scramble=True, seed=self.rng.integers(2 ** 31))
        pert = lb + (ub - lb) * sob.random(self.n_cand)
        prob = min(20.0 / self.dim, 1.0)
        mask = self.rng.random((self.n_cand, self.dim)) <= prob
        none = ~mask.any(axis=1)
        mask[none, self.rng.integers(0, self.dim, none.sum())] = True
        cand = np.where(mask, pert, center)
        with torch.no_grad():
            post = model.posterior(torch.tensor(cand))
            torch.manual_seed(int(self.rng.integers(2 ** 31)))
            s = post.rsample(torch.Size([1])).squeeze().numpy()
        return cand[int(np.argmin(s))]

    # ---- interface ----
    def ask(self, n):
        out = []
        for _ in range(n):
            if self.init_queue:
                u = np.array(self.init_queue.pop(0))
            elif len(self.y) - self.restart_from < 2:
                u = self.rng.random(self.dim)
            else:
                u = self._propose()
            self.pending.append(u.tolist())
            out.append(self._from_unit(u))
        return out

    def tell(self, X, y, g):
        if len(X) > 1 and not self.pending:
            # warm start: prior data replaces the initial design
            for x, v in zip(X, y):
                v = float(v)
                if not np.isfinite(v):
                    v = max(self.y) if self.y else 1e6
                self.X.append(self._to_unit(x).tolist())
                self.y.append(v)
            self.restart_from = 0
            self.init_queue = []
            self.run_init = 0
            self.best_run = min(self.y)
            return
        for x, v in zip(X, y):
            u = self._to_unit(x)
            # drop the matching pending point, if any
            for i, p in enumerate(self.pending):
                if np.allclose(p, u, atol=1e-9):
                    self.pending.pop(i)
                    break
            v = float(v)
            if not np.isfinite(v):
                v = max(self.y) if self.y else 1e6
            in_init = len(self.y) - self.restart_from < self.run_init
            self.X.append(u.tolist())
            self.y.append(v)
            if in_init:
                self.best_run = v if self.best_run is None else min(self.best_run, v)
                continue
            if self.best_run is None or v < self.best_run - 1e-3 * abs(self.best_run):
                self.succ += 1
                self.fail = 0
            else:
                self.succ = 0
                self.fail += 1
            self.best_run = v if self.best_run is None else min(self.best_run, v)
            if self.succ >= self.succ_tol:
                self.length = min(2 * self.length, self.length_max)
                self.succ = 0
            elif self.fail >= self.fail_tol:
                self.length /= 2
                self.fail = 0
            if self.length < self.length_min:
                self.n_restarts += 1
                self._new_run()

    def save(self, path):
        d = {k: v for k, v in self.__dict__.items() if k not in ("rng", "lo", "hi")}
        d["lo"] = self.lo.tolist()
        d["hi"] = self.hi.tolist()
        d["rng"] = self.rng.bit_generator.state
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        obj = cls.__new__(cls)
        rng_state = d.pop("rng")
        obj.__dict__.update(d)
        obj.lo = np.array(d["lo"])
        obj.hi = np.array(d["hi"])
        obj.rng = np.random.default_rng()
        obj.rng.bit_generator.state = rng_state
        return obj
