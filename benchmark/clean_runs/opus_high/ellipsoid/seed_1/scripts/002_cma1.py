import pickle
import numpy as np
import cma


class Optimizer:
    """CMA-ES with IPOP-style restarts, served one point at a time.

    Starts at the best told point (or the box centre); on stop, restarts with doubled
    population, alternating between the best point so far and a uniform random mean.
    """

    def __init__(self, space, seed=0, sigma0=2.0, popsize=None):
        self.bounds = np.array(space["bounds"], dtype=float)
        self.seed = int(seed)
        self.sigma0 = float(sigma0)
        self.popsize = popsize
        self.rng = np.random.default_rng(self.seed)
        self.es = None
        self.restarts = 0
        self.pop = []        # current population (to ask)
        self.next = 0        # index of next population member to hand out
        self.results = {}    # population index -> value
        self.asked = {}      # tuple(x) -> population index
        self.best_x = None
        self.best_y = np.inf

    def _new_es(self, mean):
        opts = {"bounds": [self.bounds[:, 0].tolist(), self.bounds[:, 1].tolist()],
                "verbose": -9, "seed": int(self.rng.integers(1, 2**31 - 1))}
        if self.popsize:
            opts["popsize"] = int(self.popsize)
        self.es = cma.CMAEvolutionStrategy(list(mean), self.sigma0, opts)
        self.popsize = self.es.popsize

    def _new_pop(self):
        if self.es is None:
            mean = self.best_x if self.best_x is not None else self.bounds.mean(axis=1)
            self._new_es(mean)
        self.pop = [np.clip(p, self.bounds[:, 0], self.bounds[:, 1]) for p in self.es.ask()]
        self.next = 0
        self.results = {}
        self.asked = {}

    def ask(self, n):
        out = []
        for _ in range(n):
            if self.next >= len(self.pop):
                if len(self.results) < len(self.pop):
                    break  # wait for outstanding values
                self._new_pop()
            x = self.pop[self.next]
            self.asked[tuple(np.round(x, 12))] = self.next
            self.next += 1
            out.append(x.tolist())
        return out

    def tell(self, X, y, g):
        for x, v in zip(X, y):
            x = np.asarray(x, dtype=float)
            v = float(v)
            if v < self.best_y:
                self.best_y, self.best_x = v, x.copy()
            k = self.asked.pop(tuple(np.round(x, 12)), None)
            if k is not None:
                self.results[k] = v
        if self.pop and len(self.results) == len(self.pop) and self.next >= len(self.pop):
            self.es.tell(self.pop, [self.results[i] for i in range(len(self.pop))])
            self.pop, self.next, self.results = [], 0, {}
            if self.es.stop():
                self.restarts += 1
                self.popsize = 2 * self.popsize
                if self.restarts % 2 == 1:
                    mean = self.best_x
                else:
                    mean = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
                self._new_es(mean)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
