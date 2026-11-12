import pickle, numpy as np, torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

def tf(y):
    y = np.asarray(y, float)
    return np.sign(y) * np.log1p(np.abs(y))

class Optimizer:
    def __init__(self, space, seed=0, n_init=0, **kw):
        self.lo = np.array([b[0] for b in space["bounds"]]); self.hi = np.array([b[1] for b in space["bounds"]])
        self.d = space["dim"]; self.seed = seed; self.n_init = n_init
        self.X = []; self.y = []; self.k = 0
        self.rng = np.random.default_rng(seed)
    def ask(self, n):
        out = []
        for _ in range(n):
            out.append(self._one())
        return out
    def _one(self):
        if len(self.X) < max(self.n_init, 6):
            return list(self.rng.uniform(self.lo, self.hi))
        X = (np.array(self.X) - self.lo) / (self.hi - self.lo)
        z = tf(self.y); m, s = z.mean(), z.std() + 1e-9
        Xt = torch.tensor(X, dtype=torch.double); Yt = torch.tensor(-(z - m) / s, dtype=torch.double)[:, None]
        try:
            gp = SingleTaskGP(Xt, Yt); fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            with torch.no_grad():
                mu = gp.posterior(Xt).mean.squeeze(-1)
            acq = LogExpectedImprovement(gp, best_f=mu.max())
            top = X[np.argsort(z)[:5]]
            g = self.rng.uniform(size=(1500, self.d))
            sig = [0.02, 0.05, 0.1, 0.2][self.k % 4]
            l = np.clip(top[self.rng.integers(0, len(top), 1500)] + self.rng.normal(0, sig, (1500, self.d)), 0, 1)
            C = np.vstack([g, l]); self.k += 1
            with torch.no_grad():
                a = acq(torch.tensor(C, dtype=torch.double)[:, None, :]).numpy()
            x = C[int(np.argmax(a))]
        except Exception:
            x = self.rng.uniform(size=self.d)
        return list(self.lo + x * (self.hi - self.lo))
    def tell(self, X, y, g):
        self.X += [list(map(float, x)) for x in X]; self.y += [float(v) for v in y]
    def save(self, path):
        pickle.dump({"s": self.__dict__}, open(path, "wb"))
    @classmethod
    def load(cls, path):
        o = cls.__new__(cls); o.__dict__.update(pickle.load(open(path, "rb"))["s"]); return o
