"""Isolation Forest (Liu, Ting & Zhou, 2008).

The idea is almost suspiciously simple: pick a random feature, cut it at a
random value, repeat. Outliers live in sparse regions so they get isolated
after only a few cuts, normal points need many. Average the path length over
a bunch of random trees and you have an anomaly score.

Trees are stored as flat arrays so scoring is a vectorised walk over all
samples at once instead of a python recursion per sample.
"""

import numpy as np

EULER_GAMMA = 0.5772156649015329


def avg_path_length(n):
    """c(n): average path length of an unsuccessful BST search over n points.

    Used both to normalise scores and to account for the subtree we didn't
    bother building below a leaf that still holds more than one point.
    """
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + EULER_GAMMA) - 2.0 * (n[big] - 1.0) / n[big]
    out[n == 2] = 1.0
    return out


class _IsoTree:
    def __init__(self, X, height_limit, rng):
        self._feat, self._thr, self._left, self._right, self._size = [], [], [], [], []
        self._grow(X, 0, height_limit, rng)
        self.feature = np.array(self._feat, dtype=np.int64)
        self.threshold = np.array(self._thr)
        self.left = np.array(self._left, dtype=np.int64)
        self.right = np.array(self._right, dtype=np.int64)
        self.size = np.array(self._size, dtype=np.int64)
        del self._feat, self._thr, self._left, self._right, self._size

    def _grow(self, X, depth, limit, rng):
        node = len(self._feat)
        self._feat.append(-1)
        self._thr.append(0.0)
        self._left.append(-1)
        self._right.append(-1)
        self._size.append(len(X))

        if depth >= limit or len(X) <= 1:
            return node

        lo, hi = X.min(axis=0), X.max(axis=0)
        splittable = np.flatnonzero(hi > lo)
        if splittable.size == 0:  # all duplicates, nothing left to isolate
            return node

        f = rng.choice(splittable)
        t = rng.uniform(lo[f], hi[f])
        goes_left = X[:, f] < t

        self._feat[node] = f
        self._thr[node] = t
        self._left[node] = self._grow(X[goes_left], depth + 1, limit, rng)
        self._right[node] = self._grow(X[~goes_left], depth + 1, limit, rng)
        return node

    def path_length(self, X):
        node = np.zeros(len(X), dtype=np.int64)
        depth = np.zeros(len(X))
        while True:
            internal = np.flatnonzero(self.feature[node] >= 0)
            if internal.size == 0:
                break
            cur = node[internal]
            go_left = X[internal, self.feature[cur]] < self.threshold[cur]
            node[internal] = np.where(go_left, self.left[cur], self.right[cur])
            depth[internal] += 1
        return depth + avg_path_length(self.size[node])


class IsolationForest:
    """Batch isolation forest.

    score_samples() returns the paper's score in (0, 1]: close to 1 means
    "very anomalous", around 0.5 or below means "nothing special". Note this
    is the opposite sign of sklearn's score_samples, which I always found
    confusing.
    """

    def __init__(self, n_estimators=100, max_samples=256, contamination=0.05, random_state=None):
        if not 0.0 < contamination < 0.5:
            raise ValueError("contamination should be in (0, 0.5)")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("expected a 2D array")
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.psi_ = min(self.max_samples, n)
        # past log2(psi) the tree is mostly isolating normal points, which we
        # don't care about - this cap is a big part of why iforest is fast
        limit = int(np.ceil(np.log2(max(self.psi_, 2))))
        self.trees_ = [
            _IsoTree(X[rng.choice(n, self.psi_, replace=False)], limit, rng)
            for _ in range(self.n_estimators)
        ]
        self.n_features_ = X.shape[1]
        self.threshold_ = np.quantile(self.score_samples(X), 1.0 - self.contamination)
        return self

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"expected shape (n, {self.n_features_})")
        mean_path = np.mean([t.path_length(X) for t in self.trees_], axis=0)
        return 2.0 ** (-mean_path / avg_path_length(self.psi_))

    def predict(self, X):
        """1 = anomaly, 0 = normal."""
        return (self.score_samples(X) > self.threshold_).astype(int)

    def fit_predict(self, X):
        return self.fit(X).predict(X)
