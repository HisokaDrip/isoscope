"""Small synthetic datasets with known outliers, used by the benchmark and tests."""

import numpy as np


def blobs(n=2000, n_features=4, outlier_frac=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_out = int(n * outlier_frac)
    centers = rng.uniform(-6, 6, size=(3, n_features))
    which = rng.integers(0, 3, n - n_out)
    inliers = centers[which] + rng.normal(scale=0.8, size=(n - n_out, n_features))
    outliers = rng.uniform(-10, 10, size=(n_out, n_features))
    return _shuffle(inliers, outliers, rng)


def donut(n=2000, outlier_frac=0.05, seed=0):
    # outliers sit in the hole in the middle - a nice case where "far from
    # the mean" is the wrong idea of anomalous
    rng = np.random.default_rng(seed)
    n_out = int(n * outlier_frac)
    theta = rng.uniform(0, 2 * np.pi, n - n_out)
    r = rng.normal(5.0, 0.3, n - n_out)
    inliers = np.c_[r * np.cos(theta), r * np.sin(theta)]
    outliers = rng.normal(0.0, 1.0, size=(n_out, 2))
    return _shuffle(inliers, outliers, rng)


def high_dim(n=3000, n_features=40, n_informative=4, outlier_frac=0.03, seed=0):
    # anomalies only differ in a few dims, the rest is noise
    rng = np.random.default_rng(seed)
    n_out = int(n * outlier_frac)
    inliers = rng.normal(size=(n - n_out, n_features))
    outliers = rng.normal(size=(n_out, n_features))
    outliers[:, :n_informative] += rng.choice([-4.0, 4.0], size=(n_out, n_informative))
    return _shuffle(inliers, outliers, rng)


def _shuffle(inliers, outliers, rng):
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(len(inliers), int), np.ones(len(outliers), int)]
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


DATASETS = {"blobs": blobs, "donut": donut, "high_dim": high_dim}
