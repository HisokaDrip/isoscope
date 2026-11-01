import time

import numpy as np

from .datasets import DATASETS
from .iforest import IsolationForest
from .metrics import roc_auc


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run(seeds=(0, 1, 2), n_estimators=100):
    try:
        from sklearn.ensemble import IsolationForest as SkIF
    except ImportError:  # benchmark still works, just without the comparison
        SkIF = None

    rows = []
    for name, make in DATASETS.items():
        ours, theirs, t_ours, t_theirs = [], [], [], []
        for seed in seeds:
            X, y = make(seed=seed)
            scores, dt = _timed(
                lambda: IsolationForest(n_estimators, random_state=seed).fit(X).score_samples(X)
            )
            ours.append(roc_auc(y, scores))
            t_ours.append(dt)
            if SkIF is not None:
                # sklearn's score_samples is "higher = more normal", flip it
                scores, dt = _timed(
                    lambda: -SkIF(n_estimators=n_estimators, random_state=seed).fit(X).score_samples(X)
                )
                theirs.append(roc_auc(y, scores))
                t_theirs.append(dt)
        X, _ = make(seed=0)
        rows.append({
            "dataset": name,
            "shape": X.shape,
            "auc": np.mean(ours),
            "auc_sklearn": np.mean(theirs) if theirs else None,
            "time": np.mean(t_ours),
            "time_sklearn": np.mean(t_theirs) if t_theirs else None,
        })
    return rows


def format_rows(rows):
    lines = [f"{'dataset':<10} {'shape':<12} {'auc':>7} {'sklearn':>8} {'time':>8} {'sklearn':>8}"]
    for r in rows:
        sk_auc = f"{r['auc_sklearn']:.4f}" if r["auc_sklearn"] is not None else "-"
        sk_t = f"{r['time_sklearn']:.2f}s" if r["time_sklearn"] is not None else "-"
        shape = "x".join(map(str, r["shape"]))
        lines.append(f"{r['dataset']:<10} {shape:<12} {r['auc']:>7.4f} {sk_auc:>8} {r['time']:>7.2f}s {sk_t:>8}")
    return "\n".join(lines)
