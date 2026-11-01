"""Turn raw HS-tree scores into yes/no alerts.

Mass scores are very heavy tailed (a few hundred for a weird point, tens of
thousands for a boring one), so we work in log space and alert when a score
drops well below the recent typical level, using median/MAD so a burst of
anomalies doesn't drag the threshold down with it.
"""

from collections import deque

import numpy as np

from .hstree import HalfSpaceTrees


class StreamDetector:
    def __init__(self, k=5.0, history=500, warmup_windows=1, min_run=2, **hst_kwargs):
        hst_kwargs.setdefault("window", 600)
        hst_kwargs.setdefault("depth", 12)
        hst_kwargs.setdefault("n_trees", 50)
        self.model = HalfSpaceTrees(**hst_kwargs)
        self.k = k
        self.history = deque(maxlen=history)
        # hold off alerting until the reference profile has seen at least one
        # full window; bump this if your first window isn't representative
        self.warmup = warmup_windows * self.model.window
        # require a couple of low ticks in a row before alerting. one weird
        # sample out of thousands is usually just noise, and it costs a
        # single tick of latency on real events
        self.min_run = min_run
        self._run = 0
        self.t = 0

    def threshold(self):
        if len(self.history) < 50:
            return None
        h = np.fromiter(self.history, float)
        med = np.median(h)
        mad = 1.4826 * np.median(np.abs(h - med)) + 1e-9
        return med - self.k * mad

    def step(self, x):
        """Returns (log_score or None, is_anomaly)."""
        raw = self.model.score_and_learn(x)
        self.t += 1
        if raw is None:
            return None, False
        s = float(np.log(raw + 1.0))
        thr = self.threshold()
        low = thr is not None and self.t > self.warmup and s < thr
        self._run = self._run + 1 if low else 0
        if not low:
            self.history.append(s)
        return s, self._run >= self.min_run
