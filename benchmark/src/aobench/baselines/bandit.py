"""An online UCB1 bandit that picks among five of the baselines, block by block.

Portfolio: CMA-ES, Sobol, COBYLA, L-BFGS-B and GP-BO, each built exactly as the
standalone baseline (same seed, and the local solvers restart when they stop).
Each arm keeps its own state and is told only its own observations. Switching
pauses an arm and later resumes it, including a half-finished CMA-ES generation
or finite-difference gradient. There is no offline training and no LLM.

Budget: first, one paid initialization block per arm, of
min(25, budget // 5) evaluations. Then the rest of the budget goes in blocks of
25, each to the arm with the highest index; the last block is cut to the
budget. At 60 evaluations the whole budget is initialization (12 per arm). An
arm that raises OptimizerExhausted retires, and its share goes to the others.

Reward: progress on a log scale, so it does not depend on the objective's
units, offset or how far the campaign has come. A block earns
log10((b0 - ref) / (b1 - ref)), where b0 and b1 are the arm's best observation
before and after the block, and ref = y(1) - (y(5) - y(1)) is a reference just
below the campaign's best: y(1) and y(5) are the best and fifth-best
observations so far, from every arm, fixed when the block starts. The spread of
the best five stands in for the unknown distance to the optimum. The reward is
rescaled to a 25-evaluation rate and clipped to [0, 1] (1 = a decade per block,
or a jump below ref). An arm's initialization block is scored once every arm has
had one: the progress of its second half over its first, against a reference
from all initialization observations. Only observed values are used: no
noise-free values, no known optimum.

Index: UCB1 (Auer, Cesa-Bianchi & Fischer 2002), with each arm's mean reward
divided by the best arm's mean, so the exploration bonus is measured against
the best arm rather than against a task's overall rate of progress:
mean / best mean + 0.25 * sqrt(2 ln(total blocks) / arm blocks), ties broken at
random with the seed. When every arm has stalled (all means 0), the bonus alone
picks, and the bandit cycles through the arms. The rewards drift as each
optimizer progresses, so the stationary-bandit regret bounds do not apply.

pycma draws from NumPy's global generator, so each arm runs with its own saved
copy of that state; the arms cannot perturb each other's random streams.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from . import OptimizerExhausted


class Bandit:
    arm_names = ("cma_es", "sobol", "cobyla", "lbfgsb", "gpbo")
    block_size = 25
    exploration = 0.25
    reference_rank = 5

    def __init__(self, bounds, make_arm, seed=None, budget=100):
        """`make_arm(name)` builds the baseline `name` for this task and seed."""
        self.bounds = np.asarray(bounds, dtype=float)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be a positive integer")
        self.dim, self.budget = len(self.bounds), budget
        self.rng = np.random.default_rng(seed)
        self._np_state = {}
        self.arms = []
        for k, name in enumerate(self.arm_names):
            with self._isolated(k):
                self.arms.append(make_arm(name))
        n = len(self.arms)
        self.rewards = [[] for _ in range(n)]
        self.arm_values = [[] for _ in range(n)]
        self.exhausted = np.zeros(n, dtype=bool)
        self.exhaustion_events, self.blocks = [], []
        self._initialized = False
        self._active = None
        self._pending = None
        self.X_, self.y_ = np.empty((0, self.dim)), np.empty(0)

    @contextmanager
    def _isolated(self, arm):
        """Run with arm's own copy of NumPy's global random state."""
        outside = np.random.get_state()
        if arm in self._np_state:
            np.random.set_state(self._np_state[arm])
        try:
            yield
        finally:
            self._np_state[arm] = np.random.get_state()
            np.random.set_state(outside)

    # ---- reward -------------------------------------------------------------------------------
    def _reference(self, values):
        """A point just below the best observation: best minus the spread of the best few."""
        s = np.sort(values)
        spread = s[min(self.reference_rank, len(s)) - 1] - s[0]
        return float(s[0] - max(spread, 1e-12 * max(1.0, abs(s[0]))))

    def _progress(self, before, after, ref, n):
        """Log10 shrink of the gap to ref over n evaluations, as a 25-evaluation rate, in [0, 1]."""
        if n < 1:
            return 0.0
        if after - ref <= 0:
            return 1.0
        decades = np.log10((before - ref) / (after - ref))
        return float(np.clip(decades * self.block_size / n, 0, 1))

    def _score_initialization(self):
        ref = self._reference(self.y_)
        for block in self.blocks:
            arm, v = block["arm"], self.arm_values[block["arm"]]
            half = max(1, len(v) // 2)
            block["reward"] = self._progress(min(v[:half]), min(v), ref, len(v) - half)
            block["reference"] = ref
            self.rewards[arm].append(block["reward"])
        self._initialized = True

    # ---- selection ----------------------------------------------------------------------------
    def _select_arm(self):
        available = np.flatnonzero(~self.exhausted)
        if not len(available):
            raise OptimizerExhausted("every bandit arm is exhausted")
        unplayed = [a for a in available if not self.arm_values[a]]
        if unplayed:
            return int(unplayed[0]), None
        means = np.array([np.mean(self.rewards[a]) for a in available])
        counts = np.array([len(self.rewards[a]) for a in available])
        index = np.full(len(self.arms), -np.inf)
        index[available] = (means / max(means.max(), 1e-12)
                            + self.exploration * np.sqrt(2 * np.log(counts.sum()) / counts))
        choices = np.flatnonzero(index == index.max())
        return int(self.rng.choice(choices)), [float(v) if np.isfinite(v) else None for v in index]

    def _start_block(self):
        arm, index = self._select_arm()
        initial = not self._initialized
        length = min(self.block_size, max(1, self.budget // len(self.arms))) if initial else self.block_size
        values = self.arm_values[arm]
        self._active = {"block": len(self.blocks) + 1, "arm": arm, "optimizer": self.arm_names[arm],
                        "after": len(self.y_), "n_points": min(length, self.budget - len(self.y_)),
                        "used": 0, "initialization": initial, "ucb_index": index,
                        "best_before": min(values) if values else None,
                        "reference": None if initial else self._reference(self.y_)}

    # ---- ask / tell ---------------------------------------------------------------------------
    def ask(self, n=1):
        if n != 1 or self._pending is not None:
            raise ValueError("the bandit needs sequential ask(1), tell calls")
        if len(self.y_) >= self.budget:
            raise OptimizerExhausted("the bandit's budget is spent")
        while True:
            if self._active is None:
                self._start_block()
            arm = self._active["arm"]
            try:
                with self._isolated(arm):
                    X = self.arms[arm].ask(1)
                break
            except OptimizerExhausted as error:
                self.exhausted[arm] = True
                self.exhaustion_events.append({"optimizer": self.arm_names[arm], "after": len(self.y_),
                                               "reason": str(error)})
                if self._active["used"]:
                    self._active["exhausted"] = True
                    self._finish_block()
                else:
                    self._active = None
                    self._maybe_score_initialization()
        self._pending = np.asarray(X, dtype=float).copy()
        return self._pending.copy()

    def tell(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if (self._pending is None or not np.array_equal(X, self._pending) or y.shape != (1,)
                or not np.isfinite(y).all()):
            raise ValueError("tell must supply the pending point and one finite observation")
        arm = self._active["arm"]
        with self._isolated(arm):
            self.arms[arm].tell(X, y)
        self.X_, self.y_ = np.vstack([self.X_, X]), np.concatenate([self.y_, y])
        self.arm_values[arm].append(float(y[0]))
        self._pending = None
        self._active["used"] += 1
        if self._active["used"] == self._active["n_points"]:
            self._finish_block()

    def _finish_block(self):
        block, arm = self._active, self._active["arm"]
        self._active = None
        block["best_after"] = min(self.arm_values[arm])
        if block["initialization"]:
            block["reward"] = None  # scored once every arm has its initialization block
            self.blocks.append(block)
            self._maybe_score_initialization()
            return
        block["reward"] = self._progress(block["best_before"], block["best_after"], block["reference"],
                                         block["used"])
        self.rewards[arm].append(block["reward"])
        self.blocks.append(block)

    def _maybe_score_initialization(self):
        if not self._initialized and all(self.arm_values[a] or self.exhausted[a] for a in range(len(self.arms))):
            self._score_initialization()

    def best(self):
        if not len(self.y_):
            raise RuntimeError("No evaluations recorded yet; call tell() first.")
        i = int(np.argmin(self.y_))
        return self.X_[i], self.y_[i]

    def metadata(self):
        return {"selector": "ucb1_bandit", "portfolio": list(self.arm_names), "block_size": self.block_size,
                "budget": self.budget, "exploration": self.exploration,
                "reward": "log10 gap shrink to best - (5th best - best), 25-eval rate, clipped [0, 1]",
                "arm_evaluations": {n: len(v) for n, v in zip(self.arm_names, self.arm_values)},
                "arm_blocks": [len(r) for r in self.rewards],
                "arm_reward_means": [float(np.mean(r)) if r else None for r in self.rewards],
                "exhaustion_events": self.exhaustion_events, "blocks": self.blocks}
