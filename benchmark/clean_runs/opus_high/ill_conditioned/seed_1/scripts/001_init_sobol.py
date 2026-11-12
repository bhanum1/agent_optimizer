import json
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Scrambled Sobol space-filling design over the encoded box."""

    def __init__(self, space, seed=0, n_total=64):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.seed = int(seed)
        self.n_total = int(n_total)
        self.next_index = 0

    def _points(self):
        s = qmc.Sobol(d=len(self.bounds), scramble=True, seed=self.seed)
        u = s.random(self.n_total)
        return self.bounds[:, 0] + u * (self.bounds[:, 1] - self.bounds[:, 0])

    def ask(self, n):
        if self.next_index >= self.n_total:
            return []
        pts = self._points()
        out = pts[self.next_index:self.next_index + n]
        self.next_index += len(out)
        return out.tolist()

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"bounds": self.bounds.tolist(), "seed": self.seed,
                       "n_total": self.n_total, "next_index": self.next_index}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        obj = cls.__new__(cls)
        obj.bounds = np.array(d["bounds"])
        obj.seed = d["seed"]
        obj.n_total = d["n_total"]
        obj.next_index = d["next_index"]
        return obj
