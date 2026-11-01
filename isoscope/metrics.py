import numpy as np


def roc_auc(y_true, scores):
    """ROC AUC via the Mann-Whitney U statistic (ties get average rank)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes for AUC")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    # average ranks over ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def event_report(flags, events, lag=10):
    """Event-level scoring for streams.

    An injected event counts as caught if anything is flagged between its
    start and end+lag (detectors need a moment). Any flag outside every
    event window is a false alarm. Point-wise precision is kind of
    meaningless for a level shift that lasts 200 ticks, hence this.
    """
    flags = np.asarray(flags, dtype=bool)
    covered = np.zeros(len(flags), dtype=bool)
    caught = []
    for ev in events:
        lo, hi = ev["start"], min(len(flags), ev["end"] + lag)
        covered[lo:hi] = True
        hit = np.flatnonzero(flags[lo:hi])
        caught.append(None if hit.size == 0 else int(lo + hit[0]))
    false_alarms = int((flags & ~covered).sum())
    return caught, false_alarms
