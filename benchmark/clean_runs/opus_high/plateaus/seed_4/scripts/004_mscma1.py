import pickle
import numpy as np
import cma


def _feats(X):
    X = np.atleast_2d(X)
    n, d = X.shape
    f = [np.ones(n)] + [X[:, i] for i in range(d)] + [X[:, i] * X[:, j] for i in range(d) for j in range(i, d)]
    return np.c_[tuple(f)]


class Optimizer:
    """Multistart local CMA-ES in the bottom of a quadratic bowl.
    A full quadratic is fit to all points with y < fit_below. Each start is a candidate with
    quadratic prediction < pred_min + delta, chosen to be farthest from every point seen with
    y < explored_below. Each local run: CMA popsize `popsize`, sigma0, at most run_evals evals;
    abandoned after abandon_after evals if its best is > global best + abandon_margin."""

    def __init__(self, space, seed=0, delta=2.0, sigma0=0.2, popsize=8, run_evals=100,
                 abandon_after=48, abandon_margin=1.5, fit_below=150.0, explored_below=67.0,
                 n_cand=4000, min_sigma=0.004):
        b = np.array(space["bounds"], dtype=float)
        self.lb, self.ub = b[:, 0], b[:, 1]
        self.dim = len(self.lb)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.delta, self.sigma0, self.popsize = delta, sigma0, popsize
        self.run_evals, self.abandon_after, self.abandon_margin = run_evals, abandon_after, abandon_margin
        self.fit_below, self.explored_below = fit_below, explored_below
        self.n_cand, self.min_sigma = n_cand, min_sigma
        self.X, self.y = [], []
        self.es = None
        self.run = 0
        self.run_n, self.run_best = 0, np.inf
        self.gen, self.gen_y, self.next_idx, self.outstanding = [], {}, 0, {}
        self.log = []

    # --- start selection
    def _start(self):
        X, y = np.array(self.X).reshape(-1, self.dim), np.array(self.y)
        if len(y) < 3 * len(_feats(self.lb)[0]):
            x0 = self.rng.uniform(self.lb, self.ub)
            self.log.append(("start", self.run, list(map(float, x0)), None, None))
            return x0
        m = y < self.fit_below
        if m.sum() < 3 * len(_feats(self.lb)[0]):
            m = y <= np.sort(y)[3 * len(_feats(self.lb)[0]) - 1]
        c = np.linalg.lstsq(_feats(X[m]), y[m], rcond=None)[0]
        g = c[1:1 + self.dim]
        H = np.zeros((self.dim, self.dim))
        k = 1 + self.dim
        for i in range(self.dim):
            for j in range(i, self.dim):
                if i == j:
                    H[i, i] = 2 * c[k]
                else:
                    H[i, j] = H[j, i] = c[k]
                k += 1
        w, V = np.linalg.eigh(H)
        w = np.maximum(w, 1e-3)
        xc = np.clip(-V @ ((V.T @ g) / w), self.lb, self.ub)
        pmin = float((_feats(xc) @ c)[0])
        # uniform in the ellipsoid pred <= pmin + delta (scaled 1.2x), then filter
        z = self.rng.standard_normal((self.n_cand, self.dim))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        z *= self.rng.random((self.n_cand, 1)) ** (1 / self.dim)
        cand = xc + 1.2 * (z * np.sqrt(2 * self.delta / w)) @ V.T
        cand = np.clip(cand, self.lb, self.ub)
        p = _feats(cand) @ c
        cand = cand[p <= pmin + self.delta]
        if len(cand) == 0:
            cand = xc[None]
        E = X[y < self.explored_below]
        if len(E):
            d = np.sqrt(((cand[:, None, :] - E[None]) ** 2).sum(-1)).min(1)
            x0 = cand[np.argmax(d)]
            dmax = float(d.max())
        else:
            x0, dmax = cand[0], np.inf
        self.log.append(("start", self.run, list(map(float, x0)), dmax, pmin))
        return x0

    def _new_es(self):
        x0 = self._start()
        opts = {"seed": int(self.seed * 1000 + self.run + 1), "popsize": self.popsize,
                "bounds": [list(self.lb), list(self.ub)], "tolx": 1e-5, "tolfun": 1e-7, "verbose": -9}
        self.es = cma.CMAEvolutionStrategy(list(x0), self.sigma0, opts)
        self.run_n, self.run_best = 0, np.inf

    def _run_done(self):
        if self.es is None:
            return True
        if self.es.stop() or self.es.sigma * np.sqrt(np.max(np.diag(self.es.C))) < self.min_sigma:
            return True
        if self.run_n >= self.run_evals:
            return True
        if self.run_n >= self.abandon_after and self.run_best > min(self.y) + self.abandon_margin:
            return True
        return False

    def _new_gen(self):
        if self._run_done():
            if self.es is not None:
                self.log.append(("end", self.run, self.run_n, float(self.run_best)))
                self.run += 1
            self._new_es()
        self.gen = [list(map(float, np.clip(p, self.lb, self.ub))) for p in self.es.ask()]
        self.gen_y, self.next_idx, self.outstanding = {}, 0, {}

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.next_idx >= len(self.gen):
                if len(self.gen_y) < len(self.gen):
                    break
                self._new_gen()
            p = self.gen[self.next_idx]
            self.outstanding.setdefault(tuple(p), []).append(self.next_idx)
            self.next_idx += 1
            out.append(p)
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            v = float(v)
            self.X.append(list(map(float, x)))
            self.y.append(v)
            k = tuple(map(float, x))
            if k in self.outstanding and self.outstanding[k]:
                self.gen_y[self.outstanding[k].pop(0)] = v
                self.run_n += 1
                self.run_best = min(self.run_best, v)
        if self.gen and len(self.gen_y) == len(self.gen):
            self.es.tell([np.array(p) for p in self.gen], [self.gen_y[i] for i in range(len(self.gen))])
            self.gen, self.gen_y, self.next_idx, self.outstanding = [], {}, 0, {}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
