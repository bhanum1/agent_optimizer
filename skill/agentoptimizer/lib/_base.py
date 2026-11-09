"""Shared base for library optimizers.

A subclass works in the unit cube and implements:

    propose() -> unit point (np.ndarray), a candidate index (int) on an indexed
                 pool, or None when it has finished
    observe(u, value)   a value for a point it proposed
    absorb(u, value)    a value for a point it did not propose (warm start);
                        default: recorded in self.U / self.y only

The base handles bounds, indexed pools (a proposed point becomes the nearest
unused candidate; a proposed index is used as is), constraint merit, records, and pickle save/load.
"""
import pickle
import warnings

import numpy as np

# Harmless library noise: Sobol balance (points drawn one at a time) and torch's jit deprecation.
warnings.filterwarnings("ignore", message=".*balance properties.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def merit(y, g):
    """y if feasible, else a penalty above every feasible value, ordered by violation."""
    violation = sum(max(v, 0.0) for v in g)
    return float(y) if violation <= 1e-10 else 1e12 + 1e6 * violation + float(np.tanh(y))


class Base:
    max_ask = None  # most points it can have outstanding (asked, not yet told); None: no limit
    def __init__(self, space, seed=0):
        self.space, self.seed = space, seed
        self.dim = space["dim"]
        bounds = np.asarray(space["bounds"], dtype=float)
        self.lo, self.hi = bounds[:, 0], bounds[:, 1]
        self.indexed = space["indexed_pool"]
        self.cands = None if space["candidates"] is None else self.to_unit(np.asarray(space["candidates"]))
        self.rng = np.random.default_rng(seed)
        self.U, self.y = np.empty((0, self.dim)), np.empty(0)
        self.used, self.pending = set(), {}

    # ---- coordinates -----------------------------------------------------------
    def to_unit(self, x):
        return (np.asarray(x, dtype=float) - self.lo) / np.where(self.hi > self.lo, self.hi - self.lo, 1)

    def from_unit(self, u):
        return self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)

    def encode_config(self, config):
        """A configuration {variable name: value} -> a point of this (encoded) space."""
        variables = self.space.get("variables")
        if not variables:
            raise ValueError("this space has no variable names; give points as lists")
        x = np.zeros(self.dim)
        for v in variables:
            value, coords = config[v["name"]], v["coords"]
            if v["type"] in ("categorical", "ordinal"):
                labels = [str(c) for c in v["choices"]]
                i = labels.index(str(value)) if str(value) in labels else \
                    [float(c) for c in v["choices"]].index(float(value))
                if v["type"] == "categorical":
                    x[coords] = 0.0
                    x[coords[i]] = 1.0
                else:
                    x[coords[0]] = i
            else:
                x[coords[0]] = np.log10(float(value)) if v.get("log") else float(value)
        return x

    def best(self):
        i = int(np.argmin(self.y))
        return self.U[i], float(self.y[i])

    # ---- protocol --------------------------------------------------------------
    def ask(self, n):
        out = []
        for _ in range(n):
            u = self.propose()
            if u is None:
                break
            if self.indexed and isinstance(u, (int, np.integer)):  # an exact candidate index
                self.pending[int(u)] = self.cands[int(u)]
                out.append(int(u))
                continue
            u = np.clip(np.asarray(u, dtype=float), 0, 1)
            if self.indexed:
                free = np.array([i for i in range(len(self.cands)) if i not in self.used and i not in self.pending])
                if not len(free):
                    break
                i = int(free[np.argmin(((self.cands[free] - u) ** 2).sum(1))])
                self.pending[i] = u
                out.append(i)
            else:
                x = self.from_unit(u)
                self.pending[tuple(x)] = u
                out.append(x.tolist())
        return out

    def tell(self, X, y, g):
        for x, yi, gi in zip(X, y, g):
            key = x if self.indexed else tuple(float(v) for v in x)
            point = self.cands[x] if self.indexed else self.to_unit(x)
            value = merit(yi, gi)
            self.U, self.y = np.vstack([self.U, point]), np.append(self.y, value)
            if self.indexed:
                self.used.add(x)
            if key in self.pending:
                self.observe(self.pending.pop(key), value)
            else:
                self.absorb(point, value)

    def observe(self, u, value):
        pass

    def absorb(self, u, value):
        pass

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
