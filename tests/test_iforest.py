import numpy as np
import pytest

from isoscope import IsolationForest
from isoscope.datasets import blobs, donut
from isoscope.iforest import avg_path_length
from isoscope.metrics import roc_auc


def test_avg_path_length_known_values():
    assert avg_path_length(1) == 0
    assert avg_path_length(2) == 1
    # c(256) from the paper is ~10.24
    assert avg_path_length(256) == pytest.approx(10.24, abs=0.01)


def test_obvious_outlier_scores_highest():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(500, 3)), [[12, 12, 12]]])
    scores = IsolationForest(random_state=0).fit(X).score_samples(X)
    assert scores.argmax() == 500
    assert scores[500] > 0.7
    assert np.median(scores) < 0.5


def test_same_seed_same_scores():
    X, _ = blobs(n=400, seed=3)
    a = IsolationForest(random_state=42).fit(X).score_samples(X)
    b = IsolationForest(random_state=42).fit(X).score_samples(X)
    np.testing.assert_array_equal(a, b)


def test_predict_respects_contamination():
    X, _ = blobs(n=1000, seed=1)
    flags = IsolationForest(contamination=0.05, random_state=0).fit_predict(X)
    assert set(np.unique(flags)) <= {0, 1}
    assert 0.03 <= flags.mean() <= 0.07


def test_duplicates_and_tiny_inputs_dont_crash():
    X = np.ones((50, 2))
    s = IsolationForest(n_estimators=10, random_state=0).fit(X).score_samples(X)
    assert np.allclose(s, s[0])
    IsolationForest(n_estimators=5, random_state=0).fit(np.array([[1.0], [2.0]]))


def test_bad_shapes():
    model = IsolationForest(n_estimators=5, random_state=0).fit(np.random.rand(30, 3))
    with pytest.raises(ValueError):
        model.score_samples(np.random.rand(5, 2))
    with pytest.raises(ValueError):
        IsolationForest().fit(np.random.rand(10))
    with pytest.raises(ValueError):
        IsolationForest(contamination=0.7)


@pytest.mark.parametrize("make", [blobs, donut])
def test_close_to_sklearn(make):
    sk = pytest.importorskip("sklearn.ensemble")
    X, y = make(seed=5)
    ours = roc_auc(y, IsolationForest(random_state=0).fit(X).score_samples(X))
    theirs = roc_auc(y, -sk.IsolationForest(random_state=0).fit(X).score_samples(X))
    assert ours > 0.9
    assert abs(ours - theirs) < 0.02
