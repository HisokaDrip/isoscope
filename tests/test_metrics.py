import numpy as np
import pytest

from isoscope.metrics import event_report, roc_auc


def test_auc_matches_sklearn_with_ties():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)  # rounding forces lots of ties
    assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))


def test_auc_edges():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.3, 0.4])


def test_event_report():
    flags = np.zeros(100, bool)
    flags[[12, 50, 90]] = True
    events = [{"start": 10, "end": 15}, {"start": 30, "end": 35}]
    caught, fa = event_report(flags, events, lag=5)
    assert caught == [12, None]
    assert fa == 2
