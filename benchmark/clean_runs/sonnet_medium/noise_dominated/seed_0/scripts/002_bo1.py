import json, numpy as np, torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
import warnings; warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, space, seed=0, n_init=0, **kw):
        self.b = np.array(space["bounds"], float); self.d = len(self.b)
        self.seed = seed; self.n_init = n_init
        self.X = []; self.y = []; self.k = 0
    def _u(self, x): return (np.array(x) - self.b[:, 0]) / (self.b[:, 1] - self.b[:, 0])
    def ask(self, n):
        rng = np.random.default_rng(self.seed * 100003 + self.k + len(self.y)); self.k += 1
        if len(self.y) < max(self.n_init, 10):
            return [list(self.b[:, 0] + rng.random(self.d) * (self.b[:, 1] - self.b[:, 0])) for _ in range(n)]
        torch.manual_seed(int(rng.integers(1 << 30)))
        X = torch.tensor(np.array([self._u(x) for x in self.X]), dtype=torch.double)
        y = np.log(np.array(self.y)); y = (y - y.mean()) / (y.std() + 1e-9)
        Y = torch.tensor(-y, dtype=torch.double)[:, None]
        try:
            m = SingleTaskGP(X, Y); fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
            acq = qLogNoisyExpectedImprovement(m, X, prune_baseline=True)
            c, _ = optimize_acqf(acq, torch.stack([torch.zeros(self.d), torch.ones(self.d)]).double(),
                                 q=1, num_restarts=8, raw_samples=256)
            u = c[0].numpy()
        except Exception:
            u = rng.random(self.d)
        return [list(self.b[:, 0] + np.clip(u, 0, 1) * (self.b[:, 1] - self.b[:, 0]))] * 1 if n == 1 else [list(self.b[:, 0] + np.clip(u, 0, 1) * (self.b[:, 1] - self.b[:, 0]))] + [list(self.b[:, 0] + rng.random(self.d) * (self.b[:, 1] - self.b[:, 0])) for _ in range(n - 1)]
    def tell(self, X, y, g):
        for x, v in zip(X, y): self.X.append(list(x)); self.y.append(float(v))
    def save(self, path):
        json.dump({"seed": self.seed, "n_init": self.n_init, "X": self.X, "y": self.y, "k": self.k}, open(path, "w"))
    @classmethod
    def load(cls, path):
        s = json.load(open(path)); o = cls({"bounds": [[-5, 5]] * 5}, seed=s["seed"], n_init=s["n_init"])
        o.X, o.y, o.k = s["X"], s["y"], s["k"]; return o
