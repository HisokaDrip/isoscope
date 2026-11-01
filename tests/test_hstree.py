import numpy as np

from isoscope import HalfSpaceTrees


def _feed(model, X):
    return [model.score_and_learn(x) for x in X]


def test_no_score_until_first_window():
    m = HalfSpaceTrees(window=50, random_state=0)
    scores = _feed(m, np.random.default_rng(0).normal(size=(60, 2)))
    assert all(s is None for s in scores[:50])
    assert all(s is not None for s in scores[50:])


def test_far_point_scores_low():
    rng = np.random.default_rng(1)
    m = HalfSpaceTrees(window=200, depth=10, random_state=1)
    _feed(m, rng.normal(size=(400, 3)))
    normal = np.median([m.score(x) for x in rng.normal(size=(50, 3))])
    weird = m.score(np.array([4.0, -4.0, 4.0]))
    assert weird < normal / 10


def test_mass_profiles_add_up():
    m = HalfSpaceTrees(n_trees=5, depth=6, window=100, random_state=0)
    _feed(m, np.random.default_rng(0).uniform(size=(130, 2)))
    # every point passes through the root of every tree exactly once
    assert np.all(m.ref[:, 0] == 100)
    assert np.all(m.latest[:, 0] == 30)
    # and child masses sum to their parent's
    for node in range(2 ** 6 - 1):
        np.testing.assert_array_equal(m.ref[:, 2 * node + 1] + m.ref[:, 2 * node + 2], m.ref[:, node])


def test_adapts_to_new_normal():
    # after a permanent shift, two windows later the shifted data is boring again
    rng = np.random.default_rng(2)
    m = HalfSpaceTrees(window=200, depth=8, random_state=2)
    _feed(m, rng.normal(size=(400, 2)))
    shifted = rng.normal(size=(600, 2)) + [1.0, 1.0]
    probe = np.array([1.0, 1.0])
    before = m.score(probe)
    _feed(m, shifted)
    assert m.score(probe) > before
