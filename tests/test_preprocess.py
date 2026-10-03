"""Unit and integration tests for data cleaning and scaling module (preprocess.py).

All unit tests use synthetic data to ensure clean runs in clones without dataset files.
One integration test runs on the real dataset and is skipped if the CSV is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.preprocess import apply_scaler, clean_missing, fit_scaler


@pytest.fixture
def sample_cfg() -> dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "paths": {
            "data_raw": "data/",
            "results": "results/",
        },
        "data": {
            "filename": "transaction_dataset.csv",
            "label_column": "FLAG",
            "suspicious_label": 1,
            "id_columns": ["Unnamed: 0", "Index", "Address"],
            "categorical_columns": [
                "erc20_most_sent_token_type",
                "erc20_most_rec_token_type",
            ],
        },
        "cleaning": {
            "policy": "drop_numeric_missing",
            "deduplicate": False,
        },
        "scaling": {
            "method": "zscore",
        },
        "paper_reference": {
            "rows": 9841,
            "benign": 7663,
            "suspicious": 2178,
            "features_text": 49,
            "features_table_a2": 45,
            "cleaned_rows": 8990,
            "cleaned_benign": 7662,
            "cleaned_suspicious": 1328,
            "dropped_rows": 851,
        },
    }


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Synthetic feature matrix X, labels y, and meta for testing clean_missing."""
    X = pd.DataFrame(
        {
            "sent_tnx": [10.0, 20.0, np.nan, 30.0, np.nan],
            "received_tnx": [5.0, 10.0, np.nan, 15.0, np.nan],
            "erc20_most_sent_token_type": ["EOS", np.nan, "BAT", " ", np.nan],
            "erc20_most_rec_token_type": ["OmiseGO", np.nan, "0", "EOS", np.nan],
        },
        index=[100, 101, 102, 103, 104],
    )
    # Row 100: clean
    # Row 101: NaN ONLY in token columns -> must be KEPT
    # Row 102: NaN in numeric columns -> must be DROPPED
    # Row 103: clean (blank string in token, not NaN) -> must be KEPT
    # Row 104: NaN in both numeric and token -> must be DROPPED

    y = pd.Series([0, 0, 1, 0, 1], index=[100, 101, 102, 103, 104], name="flag")
    meta = pd.DataFrame(
        {"address": ["0x1", "0x2", "0x3", "0x4", "0x5"]},
        index=[100, 101, 102, 103, 104],
    )
    return X, y, meta


def test_clean_missing_drops_only_numeric_nan_and_preserves_token_nan(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    sample_cfg: dict[str, Any],
) -> None:
    """Test that clean_missing drops only rows with numeric NaNs and keeps rows with token NaNs."""
    X, y, meta = synthetic_data
    X_orig = X.copy()
    y_orig = y.copy()
    meta_orig = meta.copy()

    X_clean, y_clean, meta_clean, report = clean_missing(X, y, meta, sample_cfg)

    # Check input immutability
    pd.testing.assert_frame_equal(X, X_orig)
    pd.testing.assert_series_equal(y, y_orig)
    pd.testing.assert_frame_equal(meta, meta_orig)

    # Expected kept indices: 100, 101, 103
    assert list(X_clean.index) == [100, 101, 103]
    assert list(y_clean.index) == [100, 101, 103]
    assert list(meta_clean.index) == [100, 101, 103]

    # Verify Row 101 (which has NaNs in token columns) was preserved
    assert pd.isna(X_clean.loc[101, "erc20_most_sent_token_type"])
    assert pd.isna(X_clean.loc[101, "erc20_most_rec_token_type"])

    # Verify no NaNs in numeric columns of X_clean
    assert not X_clean[["sent_tnx", "received_tnx"]].isna().any().any()

    # Report verification
    assert report["rows_before"] == 5
    assert report["rows_after"] == 3
    assert report["dropped_rows_total"] == 2
    assert report["dropped_rows_by_class"] == {0: 0, 1: 2}
    assert report["class_counts_before"] == {0: 3, 1: 2}
    assert report["class_counts_after"] == {0: 3, 1: 0}
    assert report["is_all_or_nothing_missing"] is True
    assert set(report["numeric_columns_with_nan"]) == {"sent_tnx", "received_tnx"}


def test_clean_missing_reports_paper_mismatches_without_raising(
    synthetic_data: tuple[pd.DataFrame, pd.Series, pd.DataFrame],
    sample_cfg: dict[str, Any],
) -> None:
    """Test that clean_missing populates paper_comparison without raising on mismatches."""
    X, y, meta = synthetic_data
    _, _, _, report = clean_missing(X, y, meta, sample_cfg)

    comp = report["paper_comparison"]
    assert comp["cleaned_rows"]["paper"] == 8990
    assert comp["cleaned_rows"]["actual"] == 3
    assert comp["cleaned_rows"]["matches"] is False
    assert comp["cleaned_rows"]["delta"] == 3 - 8990

    assert comp["cleaned_benign"]["paper"] == 7662
    assert comp["cleaned_benign"]["actual"] == 3
    assert comp["cleaned_benign"]["matches"] is False

    assert comp["cleaned_suspicious"]["paper"] == 1328
    assert comp["cleaned_suspicious"]["actual"] == 0
    assert comp["cleaned_suspicious"]["matches"] is False

    assert comp["dropped_rows"]["paper"] == 851
    assert comp["dropped_rows"]["actual"] == 2
    assert comp["dropped_rows"]["matches"] is False


def test_scaler_mean_std_and_constant_columns() -> None:
    """Test that fit_scaler and apply_scaler produce mean~0, std~1 and zero out constant columns without NaNs."""
    df = pd.DataFrame(
        {
            "f1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "f2": [100.0, 200.0, 300.0, 400.0, 500.0],
            "constant_col": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=[10, 20, 30, 40, 50],
    )
    df_orig = df.copy()

    scaler = fit_scaler(df)
    scaled_df = apply_scaler(scaler, df)

    # Input immutability
    pd.testing.assert_frame_equal(df, df_orig)

    # Structure preservation
    assert list(scaled_df.index) == list(df.index)
    assert list(scaled_df.columns) == list(df.columns)

    # No NaNs anywhere
    assert not scaled_df.isna().any().any()

    # Non-constant columns have mean ~ 0 and std ~ 1 (ddof=0)
    for col in ["f1", "f2"]:
        assert np.isclose(scaled_df[col].mean(), 0.0, atol=1e-12)
        assert np.isclose(scaled_df[col].std(ddof=0), 1.0, atol=1e-12)

    # Constant column scales to exactly 0.0
    assert (scaled_df["constant_col"] == 0.0).all()


def test_apply_scaler_uses_fitted_statistics_on_new_data() -> None:
    """Test that apply_scaler transforms new data using the fitted train statistics."""
    train_df = pd.DataFrame({"feat": [0.0, 10.0]})  # mean=5.0, std=5.0
    test_df = pd.DataFrame({"feat": [5.0, 15.0, 25.0]})

    scaler = fit_scaler(train_df)
    scaled_test = apply_scaler(scaler, test_df)

    # (5 - 5) / 5 = 0; (15 - 5) / 5 = 2; (25 - 5) / 5 = 4
    expected = [0.0, 2.0, 4.0]
    np.testing.assert_allclose(scaled_test["feat"].tolist(), expected)


@pytest.mark.skipif(
    not Path("data/transaction_dataset.csv").exists(),
    reason="Raw dataset CSV not present",
)
def test_real_data_pipeline(sample_cfg: dict[str, Any]) -> None:
    """Integration test running real data cleaning and scaling pipeline."""
    from src.data import TokenFrequencyEncoder, load_raw, split_columns

    df, _ = load_raw(sample_cfg)
    X, y, meta = split_columns(df, sample_cfg)

    X_clean, y_clean, meta_clean, report = clean_missing(X, y, meta, sample_cfg)

    # Assert 9,012 cleaned rows: 7,662 benign, 1,350 suspicious (829 dropped)
    assert len(X_clean) == 9012
    assert len(y_clean) == 9012
    assert len(meta_clean) == 9012
    assert report["rows_after"] == 9012
    assert report["dropped_rows_total"] == 829
    assert report["dropped_rows_by_class"] == {0: 0, 1: 829}
    assert report["class_counts_after"] == {0: 7662, 1: 1350}
    assert report["paper_comparison"]["cleaned_benign"]["matches"] is True
    assert report["paper_comparison"]["cleaned_suspicious"]["delta"] == 22

    # Encode token columns
    encoder = TokenFrequencyEncoder(columns=sample_cfg["data"]["categorical_columns"])
    X_clean_enc = encoder.fit_transform(X_clean)

    # Fit and apply scaler
    scaler = fit_scaler(X_clean_enc)
    X_scaled = apply_scaler(scaler, X_clean_enc)

    assert X_scaled.shape == (9012, 47)
    assert not X_scaled.isna().any().any()
