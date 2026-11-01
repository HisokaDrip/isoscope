"""Streaming Half-Space Trees (Tan, Ting & Liu, 2011).

Isolation forest needs the whole dataset up front. HS-trees are the
streaming cousin: the trees are built *before* seeing any data (random
splits of the unit hypercube), and all the learning is just counting how
many points land in each node.

Two mass profiles per node:
  ref    - counts from the previous window, used for scoring
  latest - counts from the current window, being filled in
When a window ends, latest becomes ref and latest is zeroed. So the model
forgets anything older than two windows, which is the point - it adapts to
drift, and a level shift stops being "anomalous" once it's the new normal.

Everything is stored as (n_trees, n_nodes) arrays in heap layout
(children of i are 2i+1, 2i+2) so one update is `depth` numpy ops across
all trees at once instead of a python loop over trees.
"""

import numpy as np


class HalfSpaceTrees:
    def __init__(self, n_trees=25, depth=10, window=250, size_limit=None, random_state=None):
        self.n_trees = n_trees
        self.depth = depth
        self.window = window
        # paper suggests 0.1 * window; stop descending once a node is this sparse
        self.size_limit = 0.1 * window if size_limit is None else size_limit
        self.rng = np.random.default_rng(random_state)
        self._buffer = []  # first window, used to learn feature ranges
        self._seen = 0
        self.lo = self.hi = None

    @property
    def ready(self):
        return self.lo is not None

    def _build(self, n_features):
        n_nodes = 2 ** (self.depth + 1) - 1
        T = self.n_trees
        self.feature = np.zeros((T, n_nodes), dtype=np.int64)
        self.split = np.zeros((T, n_nodes))
        self.ref = np.zeros((T, n_nodes))
        self.latest = np.zeros((T, n_nodes))

        for t in range(T):
            # random workspace: each dim gets a range that's guaranteed to
            # cover [0, 1] but is centred somewhere random, so splits vary
            s = self.rng.uniform(0, 1, n_features)
            half = 2 * np.maximum(s, 1 - s)
            mins, maxs = s - half, s + half
            self._grow(t, 0, mins, maxs)

    def _grow(self, t, node, mins, maxs):
        first_leaf = 2 ** self.depth - 1
        if node >= first_leaf:
            return
        q = self.rng.integers(len(mins))
        p = (mins[q] + maxs[q]) / 2.0
        self.feature[t, node] = q
        self.split[t, node] = p
        left_max = maxs.copy()
        left_max[q] = p
        right_min = mins.copy()
        right_min[q] = p
        self._grow(t, 2 * node + 1, mins, left_max)
        self._grow(t, 2 * node + 2, right_min, maxs)

    def _scale(self, x):
        return np.clip((x - self.lo) / (self.hi - self.lo), -0.5, 1.5)

    def _paths(self, x):
        """Node index at every level for every tree, shape (n_trees, depth+1)."""
        T = self.n_trees
        rows = np.arange(T)
        node = np.zeros(T, dtype=np.int64)
        path = np.empty((T, self.depth + 1), dtype=np.int64)
        path[:, 0] = 0
        for d in range(self.depth):
            go_right = x[self.feature[rows, node]] >= self.split[rows, node]
            node = 2 * node + 1 + go_right
            path[:, d + 1] = node
        return path

    def _score_scaled(self, x):
        path = self._paths(x)
        rows = np.arange(self.n_trees)[:, None]
        mass = self.ref[rows, path]
        # stop at the first node that's too sparse (or the leaf)
        sparse = mass <= self.size_limit
        sparse[:, -1] = True
        level = sparse.argmax(axis=1)
        m = mass[np.arange(self.n_trees), level]
        return float(np.sum(m * 2.0 ** level))

    def _update_scaled(self, x):
        path = self._paths(x)
        rows = np.repeat(np.arange(self.n_trees), path.shape[1])
        np.add.at(self.latest, (rows, path.ravel()), 1)
        self._seen += 1
        if self._seen % self.window == 0:
            self.ref, self.latest = self.latest, np.zeros_like(self.latest)

    def score(self, x):
        """Mass-based score: LOW means anomalous. None until the first window is in."""
        if not self.ready:
            return None
        return self._score_scaled(self._scale(np.asarray(x, dtype=float)))

    def learn(self, x):
        x = np.asarray(x, dtype=float)
        if self.ready:
            self._update_scaled(self._scale(x))
            return
        self._buffer.append(x)
        if len(self._buffer) == self.window:
            warm = np.array(self._buffer)
            lo, hi = warm.min(axis=0), warm.max(axis=0)
            pad = np.where(hi > lo, (hi - lo) * 0.1, 1.0)
            self.lo, self.hi = lo - pad, hi + pad
            self._build(warm.shape[1])
            for row in warm:
                self._update_scaled(self._scale(row))
            self._buffer = []

    def score_and_learn(self, x):
        # score first so a point can't vouch for itself
        s = self.score(x)
        self.learn(x)
        return s
