"""The optimizer behind a `points` proposal: the agent's own list of points, then finished.

`aopt commit` copies this file to blocks/ like any agent-written optimizer and
builds it with `points`, the listed configurations already encoded. It is
standalone (copied into blocks/ and loaded by host.py), so it imports nothing
from aoptcore.
"""
import pickle


class Optimizer:
    batch = True  # the whole list is requested at once

    def __init__(self, space, points=()):
        self.points = [p if isinstance(p, int) else list(p) for p in points]

    def ask(self, n):
        out, self.points = self.points[:n], self.points[n:]
        return out

    def tell(self, X, y, g):
        pass

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
