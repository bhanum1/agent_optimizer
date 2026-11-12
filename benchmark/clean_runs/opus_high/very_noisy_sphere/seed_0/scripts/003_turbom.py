"""Multi-trust-region BO (TuRBO-m style) for exploring several basins.

Keeps up to `n_tr` trust regions, served round-robin. Each ask fits a local
GP (Matern 5/2, ARD, log-shifted standardized y) on the points near its
region and maximizes LogEI inside it. A region is killed when its length
collapses below `length_min`, or when after `n_min` points its best is worse
than the global best by more than `kill_margin`. Killed centres become taboo
(radius `taboo_r`, unit-box L2; `taboo_init_r` for the given `taboo`). New regions are spawned where a global GP's
LCB is lowest, at least `spawn_r` from taboo centres and the other regions. `taboo`
gives centres (encoded units) to avoid from the start.
"""
import json
import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")


class Optimizer:
    max_ask = 1

    def __init__(self, space, seed=0, n_tr=4, length_init=0.25, length_min=0.02,
                 length_max=0.6, succ_tol=3, fail_tol=6, n_min=30,
                 kill_margin=1.5, taboo=None, taboo_init_r=0.25, taboo_r=0.12, spawn_r=0.25, beta=2.0,
                 n_local=120, max_global=800):
        self.lb = [float(b[0]) for b in space["bounds"]]
        self.ub = [float(b[1]) for b in space["bounds"]]
        self.d = len(self.lb)
        self.seed = int(seed)
        self.n_tr = int(n_tr)
        self.length_init = float(length_init)
        self.length_min = float(length_min)
        self.length_max = float(length_max)
        self.succ_tol = int(succ_tol)
        self.fail_tol = int(fail_tol)
        self.n_min = int(n_min)
        self.kill_margin = float(kill_margin)
        self.taboo_r = float(taboo_r)
        self.spawn_r = float(spawn_r)
        self.beta = float(beta)
        self.n_local = int(n_local)
        self.max_global = int(max_global)
        self.taboo = [self._to_unit(t).tolist() for t in (taboo or [])]
        self.taboo_rad = [float(taboo_init_r)] * len(self.taboo)
        self.X, self.y = [], []
        self.pending = []  # [x, tr_id]
        self.trs = []      # dicts: id, center, length, succ, fail, best, n
        self.dead = []
        self.next_id = 0
        self.turn = 0
        self.n_ask = 0
        self.rng_state = np.random.default_rng(self.seed).bit_generator.state

    # ---- helpers
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

    def _fit(self, X, y):
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
        return gp

    @staticmethod
    def _ls(gp):
        try:
            ls = gp.covar_module.base_kernel.lengthscale
        except AttributeError:
            ls = gp.covar_module.lengthscale
        return ls.detach().numpy().ravel()

    def _blocked(self, U, include_trs=True, r=None):
        centres = list(self.taboo)
        rad = list(self.taboo_rad)
        if include_trs:
            centres += [t["center"] for t in self.trs]
            rad += [self.taboo_r] * len(self.trs)
        if not centres:
            return np.zeros(len(U), bool)
        C = np.asarray(centres)
        R = np.maximum(np.asarray(rad), r or 0.0)
        D = np.linalg.norm(U[:, None, :] - C[None, :, :], axis=2)
        return (D < R[None, :]).any(axis=1)

    def _spawn(self, rng):
        import torch
        tr = {"id": self.next_id, "length": self.length_init, "succ": 0,
              "fail": 0, "best": math.inf, "n": 0}
        self.next_id += 1
        cand = rng.random((4096, self.d))
        if len(self.y) >= 10:
            X = self._to_unit(self.X)
            y = np.asarray(self.y, float)
            # observed points outside blocked zones, and jitter around them
            free = ~self._blocked(X, r=self.spawn_r)
            if free.any():
                base = X[free][np.argsort(y[free])[:20]]
                jit = base[rng.integers(len(base), size=1024)] + \
                    0.03 * rng.standard_normal((1024, self.d))
                cand = np.vstack([cand, np.clip(jit, 0, 1)])
            if len(y) > self.max_global:
                idx = np.argsort(y)[: self.max_global // 2]
                rest = np.setdiff1d(np.arange(len(y)), idx)
                idx = np.concatenate([idx, rng.choice(rest, self.max_global // 2,
                                                      replace=False)])
                X, y = X[idx], y[idx]
            gp = self._fit(X, y)
            cand = cand[~self._blocked(cand, r=self.spawn_r)]
            if len(cand) == 0:
                cand = rng.random((256, self.d))
            with torch.no_grad():
                post = gp.posterior(torch.tensor(cand, dtype=torch.double))
                lcb = post.mean.squeeze(-1) - self.beta * post.variance.squeeze(-1).sqrt()
            c = cand[int(torch.argmin(lcb))]
        else:
            c = cand[0]
        tr["center"] = c.tolist()
        self.trs.append(tr)
        return tr

    # ---- interface
    def ask(self, n):
        return [self._ask_one() for _ in range(n)]

    def _ask_one(self):
        import torch
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        rng = self._rng()
        self.n_ask += 1
        while len(self.trs) < self.n_tr:
            self._spawn(rng)
        tr = self.trs[self.turn % len(self.trs)]
        self.turn += 1
        c = np.asarray(tr["center"])
        X = self._to_unit(self.X) if self.X else np.zeros((0, self.d))
        y = np.asarray(self.y, float)
        if len(y) >= 5:
            dist = np.abs(X - c).max(axis=1)
            idx = np.argsort(dist)[: self.n_local]
            gp = self._fit(X[idx], y[idx])
            w = self._ls(gp)
            w = w / np.prod(w) ** (1.0 / self.d)
            w = np.clip(w, 0.5, 2.0)
            Xl = torch.tensor(X[idx], dtype=torch.double)
            with torch.no_grad():
                mu = gp.posterior(Xl).mean.squeeze(-1).numpy()
            # recentre on the lowest posterior mean inside the region
            half = tr["length"] * w / 2
            inside = (np.abs(X[idx] - c) <= half).all(axis=1)
            inside &= ~self._blocked(X[idx], include_trs=False)
            if inside.any():
                k = np.where(inside)[0][np.argmin(mu[inside])]
                c = X[idx][k]
                tr["center"] = c.tolist()
                best_f = float(mu[k])
            else:
                best_f = float(mu.min())
            lo = np.clip(c - half, 0, 1)
            hi = np.clip(c + half, 0, 1)
            acq = LogExpectedImprovement(gp, best_f=best_f, maximize=False)
            torch.manual_seed(int(rng.integers(1 << 30)))
            try:
                cand, _ = optimize_acqf(
                    acq, bounds=torch.tensor(np.vstack([lo, hi]), dtype=torch.double),
                    q=1, num_restarts=8, raw_samples=256)
                u = cand.detach().numpy().ravel()
            except Exception:
                u = lo + rng.random(self.d) * (hi - lo)
            if self._blocked(u[None, :], include_trs=False)[0]:
                # stay out of taboo basins: best LogEI among free samples
                S = lo + rng.random((2048, self.d)) * (hi - lo)
                S = S[~self._blocked(S, include_trs=False)]
                if len(S):
                    with torch.no_grad():
                        a = acq(torch.tensor(S, dtype=torch.double).unsqueeze(1))
                    u = S[int(torch.argmax(a))]
        else:
            u = np.clip(c + tr["length"] * (rng.random(self.d) - 0.5), 0, 1)
        x = self._from_unit(u).tolist()
        self.rng_state = rng.bit_generator.state
        self.pending.append([x, tr["id"]])
        return x

    def _kill(self, tr, why):
        self.trs = [t for t in self.trs if t["id"] != tr["id"]]
        if why != "merged":
            self.taboo.append(tr["center"])
            self.taboo_rad.append(self.taboo_r)
        self.dead.append({"id": tr["id"], "center": tr["center"],
                          "best": tr["best"], "n": tr["n"], "why": why})

    def tell(self, X, y, g=None):
        for xi, yi in zip(X, y):
            xi = [float(v) for v in xi]
            yi = float(yi)
            if not math.isfinite(yi):
                yi = max(self.y) if self.y else 1e9
            tid = None
            for k, (p, t) in enumerate(self.pending):
                if np.allclose(p, xi, atol=1e-9):
                    self.pending.pop(k)
                    tid = t
                    break
            self.X.append(xi)
            self.y.append(yi)
            tr = next((t for t in self.trs if t["id"] == tid), None)
            if tr is None:
                continue
            tr["n"] += 1
            if not math.isfinite(tr["best"]) or \
                    yi < tr["best"] - 1e-3 * abs(tr["best"]):
                tr["succ"] += 1
                tr["fail"] = 0
            else:
                tr["fail"] += 1
                tr["succ"] = 0
            tr["best"] = min(tr["best"], yi)
            if tr["succ"] >= self.succ_tol:
                tr["length"] = min(2 * tr["length"], self.length_max)
                tr["succ"] = 0
            elif tr["fail"] >= self.fail_tol:
                tr["length"] /= 2
                tr["fail"] = 0
            near = self._blocked(np.asarray([tr["center"]]), include_trs=False)[0]
            if tr["length"] < self.length_min:
                self._kill(tr, "collapsed")
            elif near:
                self._kill(tr, "merged")
            elif tr["n"] >= self.n_min and \
                    tr["best"] > min(self.y) + self.kill_margin:
                self._kill(tr, "behind")

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
