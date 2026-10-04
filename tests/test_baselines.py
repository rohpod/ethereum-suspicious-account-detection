"""Unit tests for CART and LOF baselines and inference wrappers (Phase 1.7).

All tests use synthetic data only; no raw dataset required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier

from src.baselines import compute_deltas_to_table10
from src.models import build_cart, build_lof, fit_predict_scores


def test_build_cart_defaults_and_overrides() -> None:
    """Verify build_cart instantiates DecisionTreeClassifier with expected defaults and overrides."""
    # Defaults
    cart_default = build_cart()
    assert isinstance(cart_default, DecisionTreeClassifier)
    assert cart_default.criterion == "gini"
    assert cart_default.max_depth is None
    assert cart_default.min_samples_split == 2
    assert cart_default.min_samples_leaf == 1
    assert cart_default.random_state == 42

    # Overrides
    cart_custom = build_cart(params={"max_depth": 5, "criterion": "entropy", "min_samples_split": 4})
    assert cart_custom.max_depth == 5
    assert cart_custom.criterion == "entropy"
    assert cart_custom.min_samples_split == 4


def test_build_lof_defaults_and_overrides() -> None:
    """Verify build_lof instantiates LocalOutlierFactor with novelty=True and expected defaults."""
    # Defaults
    lof_default = build_lof()
    assert isinstance(lof_default, LocalOutlierFactor)
    assert lof_default.n_neighbors == 20
    assert lof_default.contamination == "auto"
    assert lof_default.novelty is True

    # Overrides
    lof_custom = build_lof(params={"n_neighbors": 10, "contamination": 0.05})
    assert lof_custom.n_neighbors == 10
    assert lof_custom.contamination == 0.05
    assert lof_custom.novelty is True


def test_cart_fit_predict_scores_supervised() -> None:
    """Verify CART fits supervised labels, outputs binary predictions, and probability scores."""
    rng = np.random.default_rng(42)
    # Linearly separable clusters
    c0 = rng.normal(loc=0.0, scale=0.5, size=(50, 3))
    c1 = rng.normal(loc=5.0, scale=0.5, size=(50, 3))

    X_train = pd.DataFrame(np.vstack([c0, c1]), columns=["f1", "f2", "f3"])
    y_train = pd.Series([0] * 50 + [1] * 50)

    X_test = pd.DataFrame(
        np.vstack([c0[:10], c1[:10]]), columns=["f1", "f2", "f3"]
    )
    y_test = pd.Series([0] * 10 + [1] * 10)

    cart = build_cart()
    y_pred, scores = fit_predict_scores("cart", cart, X_train, y_train, X_test)

    assert len(y_pred) == len(X_test)
    assert len(scores) == len(X_test)
    assert set(np.unique(y_pred)).issubset({0, 1})
    np.testing.assert_array_equal(y_pred, y_test)
    # For class 1 samples, scores should be 1.0; for class 0, 0.0
    assert np.all(scores[10:] == 1.0)
    assert np.all(scores[:10] == 0.0)


def test_lof_fit_predict_scores_unsupervised_and_orientation() -> None:
    """Verify LOF is strictly unsupervised, ignores labels, and assigns higher scores to outliers."""
    rng = np.random.default_rng(42)
    inliers = rng.normal(loc=0.0, scale=1.0, size=(100, 2))
    outliers = rng.normal(loc=50.0, scale=1.0, size=(10, 2))

    X_train = pd.DataFrame(inliers, columns=["x", "y"])
    y_dummy = pd.Series([0] * 100)

    X_test = pd.DataFrame(np.vstack([inliers[:10], outliers]), columns=["x", "y"])

    lof = build_lof(params={"n_neighbors": 15, "contamination": 0.1})
    y_pred, scores = fit_predict_scores("lof", lof, X_train, y_dummy, X_test)

    assert len(y_pred) == len(X_test)
    assert len(scores) == len(X_test)
    assert set(np.unique(y_pred)).issubset({0, 1})

    # Outliers (index 10..19) must receive significantly higher anomaly scores than inliers (0..9)
    inlier_scores = scores[:10]
    outlier_scores = scores[10:]
    assert np.mean(outlier_scores) > np.mean(inlier_scores)
    assert np.sum(y_pred[10:]) >= 8  # Majority of obvious outliers flagged

    # Label independence test: shuffling training labels must yield identical predictions & scores
    lof_shuffled = build_lof(params={"n_neighbors": 15, "contamination": 0.1})
    y_perm = pd.Series(rng.permutation(len(X_train)))
    y_pred_perm, scores_perm = fit_predict_scores(
        "lof", lof_shuffled, X_train, y_perm, X_test
    )

    np.testing.assert_array_equal(y_pred, y_pred_perm)
    np.testing.assert_allclose(scores, scores_perm)


def test_compute_deltas_to_table10() -> None:
    """Verify compute_deltas_to_table10 calculates correct signed differences."""
    metrics = {
        "accuracy": 0.850,
        "mae": 0.150,
        "precision": 0.820,
        "recall": 0.880,
        "f1": 0.849,
        "roc_auc": 0.910,
    }
    paper_row = {
        "accuracy": 0.810,
        "mae": 0.220,
        "precision": 0.813,
        "recall": 0.810,
        "f1": 0.809,
        "auc": 0.810,
    }

    deltas = compute_deltas_to_table10(metrics, paper_row)

    assert pytest.approx(deltas["accuracy"], abs=1e-5) == 0.850 - 0.810
    assert pytest.approx(deltas["mae"], abs=1e-5) == 0.150 - 0.220
    assert pytest.approx(deltas["precision"], abs=1e-5) == 0.820 - 0.813
    assert pytest.approx(deltas["recall"], abs=1e-5) == 0.880 - 0.810
    assert pytest.approx(deltas["f1"], abs=1e-5) == 0.849 - 0.809
    assert pytest.approx(deltas["auc"], abs=1e-5) == 0.910 - 0.810


def test_input_immutability() -> None:
    """Verify that calling fit_predict_scores does not mutate feature matrices or label series."""
    df_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df_test = pd.DataFrame({"a": [7.0, 8.0], "b": [9.0, 10.0]})
    y = pd.Series([0, 1, 0])

    train_copy = df_train.copy()
    test_copy = df_test.copy()
    y_copy = y.copy()

    cart = build_cart()
    _ = fit_predict_scores("cart", cart, df_train, y, df_test)

    pd.testing.assert_frame_equal(df_train, train_copy)
    pd.testing.assert_frame_equal(df_test, test_copy)
    pd.testing.assert_series_equal(y, y_copy)
