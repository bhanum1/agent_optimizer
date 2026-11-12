import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES (pycma), points handed out one at a time from each generation.
    Starts at x0 (or the best told point if x0 is None). On a stop condition it
    restarts from the best point seen, with popsize multiplied by inc_pop (IPOP-like)."""

    def __init__(self, space, seed=0, sigma0=0.5, x0=None, popsize=None, inc_pop=2):
        self.lo = np.array([b[0] for b in space["bounds"]], float)
        self.hi = np.array([b[1] for b in space["bounds"]], float)
        self.seed, self.sigma0, self.x0, self.inc_pop = seed, sigma0, x0, inc_pop
        self.popsize = popsize
        self.es = None
        self.gen = []        # current generation (unclipped), list of arrays
        self.gen_f = {}      # index in gen -> value
        self.next_i = 0      # next index of gen to hand out
        self.asked = {}      # key of clipped point -> index in gen
        self.best_x, self.best_y = None, np.inf
        self.restarts = 0
        self.history = []    # (restart, best_y) per generation

    def _key(self, x): return tuple(np.round(np.asarray(x, float), 12))

    def _new_es(self):
        x0 = self.best_x if (self.x0 is None or self.restarts > 0) else np.asarray(self.x0, float)
        if x0 is None: x0 = (self.lo + self.hi) / 2
        opts = {"bounds": [self.lo.tolist(), self.hi.tolist()], "seed": self.seed + 1000 * self.restarts,
                "verbose": -9, "tolx": 1e-11, "tolfun": 1e-14, "tolfunhist": 1e-14}
        if self.popsize: opts["popsize"] = int(self.popsize)
        self.es = cma.CMAEvolutionStrategy(np.asarray(x0, float).tolist(), self.sigma0, opts)
        self.gen, self.gen_f, self.next_i, self.asked = [], {}, 0, {}

    def _new_gen(self):
        if self.es is None: self._new_es()
        self.gen = [np.asarray(s, float) for s in self.es.ask()]
        self.gen_f, self.next_i, self.asked = {}, 0, {}

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.es is None or self.next_i >= len(self.gen):
                if self.es is not None and len(self.gen_f) < len(self.gen):
                    break  # generation still waiting for values
                self._new_gen()
            x = np.clip(self.gen[self.next_i], self.lo, self.hi)
            self.asked[self._key(x)] = self.next_i
            self.next_i += 1
            out.append(x.tolist())
        return out

    def tell(self, X, y, g=None):
        for x, v in zip(X, y):
            v = float(v)
            if not np.isfinite(v): v = 1e300
            if v < self.best_y: self.best_y, self.best_x = v, np.asarray(x, float)
            i = self.asked.pop(self._key(x), None)
            if i is not None: self.gen_f[i] = v
        if self.es is not None and self.gen and len(self.gen_f) == len(self.gen):
            self.es.tell(self.gen, [self.gen_f[i] for i in range(len(self.gen))])
            self.history.append((self.restarts, self.best_y, float(self.es.sigma)))
            self.gen, self.gen_f, self.next_i = [], {}, 0
            if self.es.stop():
                self.restarts += 1
                lam = self.es.popsize
                self.popsize = lam * self.inc_pop
                self.es = None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls.__new__(cls)
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
