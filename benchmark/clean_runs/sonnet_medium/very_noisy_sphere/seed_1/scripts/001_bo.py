import pickle, numpy as np, torch
from scipy.stats import qmc
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class Optimizer:
    def __init__(self, space, seed=0, n_init=32, max_fit=250):
        self.space = space
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.d = len(self.lo)
        self.seed, self.n_init, self.max_fit = seed, n_init, max_fit
        self.X, self.y, self.n_asked = [], [], 0
        self.rng = np.random.RandomState(seed)

    def _u(self, X):
        return (np.asarray(X, float) - self.lo) / (self.hi - self.lo)

    def ask(self, n):
        out = []
        for _ in range(n):
            if len(self.X) < self.n_init:
                s = qmc.Sobol(self.d, seed=self.seed).random(self.n_init)
                u = s[self.n_asked % self.n_init]
            else:
                u = self._bo()
            self.n_asked += 1
            out.append((self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)).tolist())
        return out

    def _bo(self):
        X = self._u(self.X); y = np.array(self.y, float)
        if len(y) > self.max_fit:
            keep = np.unique(np.concatenate([np.argsort(y)[: self.max_fit // 2], np.arange(len(y) - self.max_fit // 2, len(y))]))
            X, y = X[keep], y[keep]
        mu, sd = y.mean(), y.std() + 1e-9
        Xt = torch.tensor(X, dtype=torch.double)
        yt = torch.tensor((y - mu) / sd, dtype=torch.double)[:, None]
        try:
            m = SingleTaskGP(Xt, yt)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
            acq = LogNoisyExpectedImprovement(m, Xt)
            torch.manual_seed(int(self.rng.randint(1 << 30)))
            c, _ = optimize_acqf(acq, torch.tensor([[0.0] * self.d, [1.0] * self.d], dtype=torch.double), q=1, num_restarts=8, raw_samples=256)
            return c[0].numpy()
        except Exception:
            return self.rng.rand(self.d)

    def tell(self, X, y, g):
        self.X += [list(map(float, x)) for x in X]
        self.y += [float(v) for v in y]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
