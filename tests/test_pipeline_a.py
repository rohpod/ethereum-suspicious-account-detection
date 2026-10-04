"""Unit tests for Pipeline A SMOTE, splitting, and deterministic data building.

All tests use synthetic data only to execute within seconds without raw data dependencies.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.pipeline_a import build_pipeline_a_data
from src.preprocess import apply_smote, split_train_test


@pytest.fixture
def pipeline_a_cfg() -> dict[str, Any]:
    """Test configuration for Pipeline A operations."""
    return {
        "seed": 42,
        "paths": {"data_raw": "data/", "results": "results/"},
        "data": {
            "filename": "transaction_dataset.csv",
            "label_column": "FLAG",
            "id_columns": ["Unnamed: 0", "Index", "Address"],
            "categorical_columns": [" ERC20 most sent token type", " ERC20_most_rec_token_type"],
        },
        "smote": {
            "k_neighbors": 5,
            "sampling_strategy": "auto",
        },
        "split": {
            "test_size": 0.2,
            "shuffle": True,
            "stratify": False,
        },
        "paper_reference": {
            "smote_rows": 15324,
            "train": 12259,
            "test": 3065,
            "test_benign": 1544,
            "test_suspicious": 1521,
        },
        "features": {
            "table2_reference": ["f_0", "f_1", "f_2"],
        },
    }


@pytest.fixture
def synthetic_imbalanced_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic imbalanced dataset: 80 benign (0), 20 suspicious (1)."""
    rng = np.random.default_rng(42)
    n_0, n_1 = 80, 20
    X_0 = rng.normal(loc=0.0, scale=1.0, size=(n_0, 4))
    X_1 = rng.normal(loc=2.0, scale=1.0, size=(n_1, 4))

    X_arr = np.vstack([X_0, X_1])
    y_arr = np.array([0] * n_0 + [1] * n_1)

    cols = [f"f_{i}" for i in range(4)]
    X = pd.DataFrame(X_arr, columns=cols)
    y = pd.Series(y_arr, name="FLAG")
    return X, y


def test_smote_balances_classes_and_appends_synthetic_rows(
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
    pipeline_a_cfg: dict[str, Any],
) -> None:
    """SMOTE must balance minority class to majority count, preserving originals first."""
    X, y = synthetic_imbalanced_data
    n_orig = len(X)

    X_res, y_res, is_synthetic, report = apply_smote(X, y, pipeline_a_cfg)

    # Balanced to 80 each -> 160 total
    assert len(X_res) == 160
    assert len(y_res) == 160
    assert (y_res == 0).sum() == 80
    assert (y_res == 1).sum() == 80

    # Original rows must occupy indices 0..n_orig-1
    np.testing.assert_array_equal(X_res.iloc[:n_orig].values, X.values)
    np.testing.assert_array_equal(y_res.iloc[:n_orig].values, y.values)

    # is_synthetic flags: False for original, True for appended
    assert not is_synthetic[:n_orig].any()
    assert is_synthetic[n_orig:].all()
    assert report["synthetic_rows_added"] == 60


def test_smote_deterministic_under_same_seed(
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
    pipeline_a_cfg: dict[str, Any],
) -> None:
    """apply_smote must produce identical arrays and flags when run with the same seed."""
    X, y = synthetic_imbalanced_data

    X_res1, y_res1, syn1, _ = apply_smote(X, y, pipeline_a_cfg)
    X_res2, y_res2, syn2, _ = apply_smote(X, y, pipeline_a_cfg)

    pd.testing.assert_frame_equal(X_res1, X_res2)
    pd.testing.assert_series_equal(y_res1, y_res2)
    np.testing.assert_array_equal(syn1, syn2)


def test_smote_synthetic_flag_count_equals_rows_added(
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
    pipeline_a_cfg: dict[str, Any],
) -> None:
    """Total True flags in is_synthetic must strictly equal rows added."""
    X, y = synthetic_imbalanced_data
    X_res, _, is_synthetic, _ = apply_smote(X, y, pipeline_a_cfg)
    assert int(np.sum(is_synthetic)) == len(X_res) - len(X)


def test_smote_accepts_dataframes_and_preserves_structure(
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
    pipeline_a_cfg: dict[str, Any],
) -> None:
    """apply_smote must accept and return pandas DataFrames with intact column names."""
    X, y = synthetic_imbalanced_data
    X_res, y_res, _, _ = apply_smote(X, y, pipeline_a_cfg)

    assert isinstance(X_res, pd.DataFrame)
    assert isinstance(y_res, pd.Series)
    assert list(X_res.columns) == list(X.columns)


def test_smote_does_not_mutate_inputs(
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
    pipeline_a_cfg: dict[str, Any],
) -> None:
    """apply_smote must not mutate the input X or y."""
    X, y = synthetic_imbalanced_data
    X_copy = X.copy()
    y_copy = y.copy()

    apply_smote(X, y, pipeline_a_cfg)

    pd.testing.assert_frame_equal(X, X_copy)
    pd.testing.assert_series_equal(y, y_copy)


def test_split_train_test_sizes_ceil_rule(pipeline_a_cfg: dict[str, Any]) -> None:
    """On n=15,324, an 80/20 train_test_split must produce exactly 12,259 and 3,065 rows."""
    n = 15324
    X_dummy = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    y_dummy = pd.Series([0] * (n // 2) + [1] * (n // 2))
    is_syn = np.zeros(n, dtype=bool)

    X_train, X_test, y_train, y_test, _, _, report = split_train_test(
        X_dummy, y_dummy, is_syn, pipeline_a_cfg
    )

    assert len(X_train) == 12259
    assert len(X_test) == 3065
    assert len(y_train) == 12259
    assert len(y_test) == 3065
    assert report["train_rows"] == 12259
    assert report["test_rows"] == 3065
    assert report["paper_comparison"]["train"]["matches"] is True
    assert report["paper_comparison"]["test"]["matches"] is True


def test_split_not_stratified(pipeline_a_cfg: dict[str, Any]) -> None:
    """With stratify=False, test set class counts reflect random draw, not forced balance."""
    n = 15324
    X_dummy = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    y_dummy = pd.Series([0] * (n // 2) + [1] * (n // 2))
    is_syn = np.zeros(n, dtype=bool)

    _, _, _, _y_test, _, _, report = split_train_test(
        X_dummy, y_dummy, is_syn, pipeline_a_cfg
    )

    counts = report["test_class_counts"]
    # With seed 42 on balanced 15,324 unstratified, counts are not 50/50 exactly
    assert counts[0] + counts[1] == 3065
    assert counts[0] != counts[1]


def test_synthetic_flags_survive_the_split(pipeline_a_cfg: dict[str, Any]) -> None:
    """Synthetic flags must partition cleanly across train and test sets."""
    n = 100
    X_dummy = pd.DataFrame(np.zeros((n, 2)), columns=["a", "b"])
    y_dummy = pd.Series([0] * 50 + [1] * 50)
    # 30 synthetic rows
    is_syn = np.zeros(n, dtype=bool)
    is_syn[70:] = True

    _, _, _, _, is_syn_tr, is_syn_te, report = split_train_test(
        X_dummy, y_dummy, is_syn, pipeline_a_cfg
    )

    assert len(is_syn_tr) == 80
    assert len(is_syn_te) == 20
    assert np.sum(is_syn_tr) + np.sum(is_syn_te) == 30
    assert report["synthetic_train_count"] + report["synthetic_test_count"] == 30


def test_split_does_not_mutate_inputs(pipeline_a_cfg: dict[str, Any]) -> None:
    """split_train_test must not mutate input X, y, or is_synthetic."""
    n = 50
    X_dummy = pd.DataFrame(np.ones((n, 2)), columns=["a", "b"])
    y_dummy = pd.Series([0] * 25 + [1] * 25)
    is_syn = np.zeros(n, dtype=bool)

    X_copy = X_dummy.copy()
    y_copy = y_dummy.copy()
    syn_copy = is_syn.copy()

    split_train_test(X_dummy, y_dummy, is_syn, pipeline_a_cfg)

    pd.testing.assert_frame_equal(X_dummy, X_copy)
    pd.testing.assert_series_equal(y_dummy, y_copy)
    np.testing.assert_array_equal(is_syn, syn_copy)


def test_build_pipeline_a_data_missing_cache_raises(
    pipeline_a_cfg: dict[str, Any],
) -> None:
    """build_pipeline_a_data must raise FileNotFoundError if PSO cache does not exist."""
    fake_cfg = copy.deepcopy(pipeline_a_cfg)
    fake_cfg["paths"]["results"] = "non_existent_dir_12345/"

    with pytest.raises(FileNotFoundError, match="PSO cache not found"):
        build_pipeline_a_data(fake_cfg, feature_set="pso")


def test_build_pipeline_a_data_no_id_columns_in_selected_features(
    pipeline_a_cfg: dict[str, Any],
) -> None:
    """build_pipeline_a_data must raise ValueError if an ID column is in features."""
    bad_cfg = copy.deepcopy(pipeline_a_cfg)
    bad_cfg["features"] = {"table2_reference": ["Address", "f_0"]}

    with pytest.raises(ValueError, match="Forbidden identifier or label column"):
        build_pipeline_a_data(bad_cfg, feature_set="table2")
