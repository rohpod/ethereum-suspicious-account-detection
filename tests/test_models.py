"""Tests for model builders and inference wrappers (src/models.py).

All tests use synthetic data only; no raw dataset required.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.models import (
    build_isolation_forest,
    build_svm,
    build_xgboost,
    fit_predict_scores,
)


def test_isolation_forest_api_facts() -> None:
    """Verify Isolation Forest API behavior, score direction, and label independence."""
    # 1. Obvious inliers around 0.0, clear outliers far away at 100.0
    rng = np.random.default_rng(42)
    inliers = rng.normal(loc=0.0, scale=1.0, size=(100, 3))
    outliers = rng.normal(loc=100.0, scale=1.0, size=(10, 3))

    X_train = pd.DataFrame(inliers, columns=["f1", "f2", "f3"])
    X_test = pd.DataFrame(np.vstack([inliers[:10], outliers]), columns=["f1", "f2", "f3"])
    y_dummy = pd.Series([0] * len(X_train))

    model = build_isolation_forest(
        params={"contamination": 0.1, "n_estimators": 50, "max_samples": "auto"}
    )
    raw_pred = model.fit(X_train).predict(X_test)

    # Scikit-learn raw predict must return only {-1, 1}
    assert set(np.unique(raw_pred)).issubset({-1, 1})

    # Wrapper fit_predict_scores must map -1 -> 1 (suspicious) and 1 -> 0 (benign)
    model_fresh = build_isolation_forest(
        params={"contamination": 0.1, "n_estimators": 50, "max_samples": "auto"}
    )
    y_pred, scores = fit_predict_scores(
        "isolation_forest", model_fresh, X_train, y_dummy, X_test
    )

    assert set(np.unique(y_pred)).issubset({0, 1})
    assert len(y_pred) == len(X_test)
    assert len(scores) == len(X_test)

    # Injected outliers (last 10) must receive higher anomaly scores than inliers (first 10)
    inlier_scores = scores[:10]
    outlier_scores = scores[10:]
    assert np.mean(outlier_scores) > np.mean(inlier_scores)

    # Inliers should mostly be classified as 0, outliers as 1
    assert np.sum(y_pred[10:]) >= 8  # Majority of obvious outliers flagged
    assert np.sum(y_pred[:10]) <= 2  # Inliers largely unflagged

    # Label independence: fitting on shuffled or inverted labels yields identical predictions
    model_shuffled = build_isolation_forest(
        params={"contamination": 0.1, "n_estimators": 50, "max_samples": "auto"}
    )
    y_shuffled = pd.Series(rng.permutation(len(X_train)))
    y_pred_shuffled, scores_shuffled = fit_predict_scores(
        "isolation_forest", model_shuffled, X_train, y_shuffled, X_test
    )
    np.testing.assert_array_equal(y_pred, y_pred_shuffled)
    np.testing.assert_allclose(scores, scores_shuffled)


def test_svm_decision_function_direction() -> None:
    """Verify SVM decision_function direction (higher for class 1) on a separable dataset."""
    rng = np.random.default_rng(42)
    # Class 0 at 0.0, Class 1 at 10.0
    c0 = rng.normal(loc=0.0, scale=0.5, size=(40, 2))
    c1 = rng.normal(loc=10.0, scale=0.5, size=(40, 2))

    X = pd.DataFrame(np.vstack([c0, c1]), columns=["x1", "x2"])
    y = pd.Series([0] * 40 + [1] * 40)

    svm = build_svm(params={"C": 1.0, "gamma": "scale"})
    y_pred, scores = fit_predict_scores("svm", svm, X, y, X)

    # Assert perfect separation and score direction
    np.testing.assert_array_equal(y_pred, y)
    assert np.mean(scores[40:]) > np.mean(scores[:40])
    assert np.min(scores[40:]) > np.max(scores[:40])


def test_xgboost_and_svm_accept_dataframe_and_deterministic() -> None:
    """Verify XGBoost and SVM accept pandas DataFrames and produce deterministic outputs."""
    df = pd.DataFrame(
        {
            "feat_a": [0.1, 0.4, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15],
            "feat_b": [1.2, 3.4, 0.5, 2.1, 0.8, 3.0, 1.1, 2.5],
        }
    )
    y = pd.Series([0, 0, 1, 0, 1, 0, 1, 0])

    for model_name, builder in [("xgboost", build_xgboost), ("svm", build_svm)]:
        model_1 = builder(params={"n_estimators": 5} if model_name == "xgboost" else {})
        pred_1, scores_1 = fit_predict_scores(model_name, model_1, df, y, df)

        assert isinstance(pred_1, np.ndarray)
        assert isinstance(scores_1, np.ndarray)
        assert len(pred_1) == len(df)
        assert len(scores_1) == len(df)
        assert set(np.unique(pred_1)).issubset({0, 1})

        # Determinism check with fresh model using same seed
        model_2 = builder(params={"n_estimators": 5} if model_name == "xgboost" else {})
        pred_2, scores_2 = fit_predict_scores(model_name, model_2, df, y, df)

        np.testing.assert_array_equal(pred_1, pred_2)
        np.testing.assert_allclose(scores_1, scores_2)


def test_model_builders_override_params() -> None:
    """Verify builder functions merge overrides properly."""
    xgb_clf = build_xgboost(params={"n_estimators": 42, "max_depth": 3})
    assert xgb_clf.get_params()["n_estimators"] == 42
    assert xgb_clf.get_params()["max_depth"] == 3

    svc = build_svm(params={"C": 5.5, "kernel": "linear"})
    assert svc.get_params()["C"] == 5.5
    assert svc.get_params()["kernel"] == "linear"

    iso = build_isolation_forest(params={"n_estimators": 25, "contamination": 0.05})
    assert iso.get_params()["n_estimators"] == 25
    assert iso.get_params()["contamination"] == 0.05


def test_svm_no_future_warning_and_score_direction() -> None:
    """Regression test: build_svm and fit raise no FutureWarning, and decision_function aligns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)

        svm = build_svm()
        rng = np.random.default_rng(42)
        c0 = rng.normal(loc=0.0, scale=0.5, size=(30, 2))
        c1 = rng.normal(loc=10.0, scale=0.5, size=(30, 2))
        X = pd.DataFrame(np.vstack([c0, c1]), columns=["x1", "x2"])
        y = pd.Series([0] * 30 + [1] * 30)

        svm.fit(X, y)
        scores = svm.decision_function(X)

    assert np.mean(scores[30:]) > np.mean(scores[:30])
    assert np.min(scores[30:]) > np.max(scores[:30])

