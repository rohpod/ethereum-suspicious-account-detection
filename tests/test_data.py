"""Unit and integration tests for data loading, splitting, encoding, and validation.

All unit tests use synthetic data to ensure tests pass in clean clones without data.
One integration test runs on the real CSV and is skipped if the file is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data import (
    TokenFrequencyEncoder,
    load_raw,
    normalise_column_name,
    split_columns,
    validate,
)


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
                " ERC20 most sent token type",
                " ERC20_most_rec_token_type",
            ],
        },
        "paper_reference": {
            "rows": 9841,
            "benign": 7663,
            "suspicious": 2178,
            "features_text": 49,
            "features_table_a2": 45,
        },
    }


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Synthetic DataFrame mimicking Ethereum transaction dataset structure."""
    data = {
        "unnamed_0": [0, 1, 2, 3, 4],
        "index": [1, 2, 3, 4, 5],
        "address": [
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "0x3333333333333333333333333333333333333333",
            "0x4444444444444444444444444444444444444444",
            "0x5555555555555555555555555555555555555555",
        ],
        "flag": [0, 0, 1, 0, 1],
        "sent_tnx": [10, 20, 5, 0, 12],
        "received_tnx": [5, 10, 2, 1, 4],
        "erc20_most_sent_token_type": ["EOS", " ", None, "0", "EOS"],
        "erc20_most_rec_token_type": [None, "OmiseGO", "0", "", "OmiseGO"],
    }
    return pd.DataFrame(data)


def test_normalise_column_name_and_uniqueness() -> None:
    """Test column normalisation logic and uniqueness validation."""
    assert normalise_column_name("  Avg min between sent tnx  ") == "avg_min_between_sent_tnx"
    assert normalise_column_name("max value received ") == "max_value_received"
    assert (
        normalise_column_name("total transactions (including tnx to create contract")
        == "total_transactions_including_tnx_to_create_contract"
    )
    assert normalise_column_name(" ERC20 uniq sent addr.1") == "erc20_uniq_sent_addr_1"
    assert normalise_column_name(" ERC20_most_rec_token_type") == "erc20_most_rec_token_type"
    assert normalise_column_name("Unnamed: 0") == "unnamed_0"

    # Test uniqueness failure on duplicate names after normalisation
    names = ["Token_Type", "Token Type", "token-type"]
    normed = [normalise_column_name(n) for n in names]
    assert len(set(normed)) == 1  # All collapse to 'token_type'


def test_split_columns_excludes_id_and_label(
    synthetic_df: pd.DataFrame, sample_cfg: dict[str, Any]
) -> None:
    """Test that split_columns properly excludes id columns and label column from X."""
    X, y, meta = split_columns(synthetic_df, sample_cfg)

    # Check X does not contain any identifier or label columns
    assert "unnamed_0" not in X.columns
    assert "index" not in X.columns
    assert "address" not in X.columns
    assert "flag" not in X.columns

    # Verify features in X
    expected_features = {
        "sent_tnx",
        "received_tnx",
        "erc20_most_sent_token_type",
        "erc20_most_rec_token_type",
    }
    assert set(X.columns) == expected_features

    # Verify meta contains id columns
    assert set(meta.columns) == {"unnamed_0", "index", "address"}

    # Verify y is binary series
    assert list(y) == [0, 0, 1, 0, 1]
    assert y.dtype == int


def test_split_columns_raises_on_forbidden_or_invalid_label(
    sample_cfg: dict[str, Any],
) -> None:
    """Test that split_columns raises if forbidden columns are in X or label is not binary."""
    # Test missing label
    bad_df = pd.DataFrame({"sent_tnx": [1, 2]})
    with pytest.raises(ValueError, match="Label column"):
        split_columns(bad_df, sample_cfg)

    # Test non-binary label
    bad_label_df = pd.DataFrame({
        "unnamed_0": [0, 1],
        "index": [1, 2],
        "address": ["0x1", "0x2"],
        "flag": [0, 2],  # Non-binary
        "sent_tnx": [1, 2],
    })
    with pytest.raises(ValueError, match="must contain binary values"):
        split_columns(bad_label_df, sample_cfg)

    # Test forbidden identifier column remaining in X
    bad_id_cfg = dict(sample_cfg)
    bad_id_cfg["data"] = dict(sample_cfg["data"])
    bad_id_cfg["data"]["id_columns"] = []  # Do not drop id columns
    with pytest.raises(ValueError, match="Forbidden identifier/label column found"):
        split_columns(
            pd.DataFrame({
                "address": ["0x1", "0x2"],
                "flag": [0, 1],
                "sent_tnx": [1, 2],
            }),
            bad_id_cfg,
        )


def test_token_frequency_encoder_categories() -> None:
    """Test TokenFrequencyEncoder handles NaN, blank, '0', and normal tokens."""
    df = pd.DataFrame({
        "erc20_most_sent_token_type": ["EOS", " ", None, "0", "EOS"],
        "erc20_most_rec_token_type": [None, "OmiseGO", "0", "", "OmiseGO"],
    })

    encoder = TokenFrequencyEncoder(columns=[
        "erc20_most_sent_token_type",
        "erc20_most_rec_token_type",
    ])
    X_enc = encoder.fit_transform(df)

    # Check columns exist and are numeric floats
    assert list(X_enc.columns) == list(df.columns)
    assert X_enc["erc20_most_sent_token_type"].dtype == float
    assert X_enc["erc20_most_rec_token_type"].dtype == float

    # Frequencies on fit: total rows = 5
    # Sent: EOS: 2/5 = 0.4; ' ' -> '__blank__': 1/5 = 0.2; None -> '__missing__': 1/5 = 0.2; '0': 1/5 = 0.2
    sent_vals = X_enc["erc20_most_sent_token_type"].tolist()
    assert np.isclose(sent_vals[0], 0.4)  # EOS
    assert np.isclose(sent_vals[1], 0.2)  # __blank__
    assert np.isclose(sent_vals[2], 0.2)  # __missing__
    assert np.isclose(sent_vals[3], 0.2)  # 0
    assert np.isclose(sent_vals[4], 0.4)  # EOS

    # Rec: None -> '__missing__': 1/5 = 0.2; OmiseGO: 2/5 = 0.4; '0': 1/5 = 0.2; '' -> '__blank__': 1/5 = 0.2
    rec_vals = X_enc["erc20_most_rec_token_type"].tolist()
    assert np.isclose(rec_vals[0], 0.2)  # __missing__
    assert np.isclose(rec_vals[1], 0.4)  # OmiseGO
    assert np.isclose(rec_vals[2], 0.2)  # 0
    assert np.isclose(rec_vals[3], 0.2)  # __blank__
    assert np.isclose(rec_vals[4], 0.4)  # OmiseGO


def test_encoder_fit_on_subset_transform_all_gives_zero_for_unseen() -> None:
    """Test that fitting on a subset yields 0.0 for unseen categories during transform."""
    train_df = pd.DataFrame({
        "token_col": ["EOS", "BAT", "EOS"],
    })
    test_df = pd.DataFrame({
        "token_col": ["EOS", "UNKNOWN_TOKEN", "ANOTHER_NEW_TOKEN"],
    })

    encoder = TokenFrequencyEncoder(columns=["token_col"])
    encoder.fit(train_df)
    transformed = encoder.transform(test_df)

    vals = transformed["token_col"].tolist()
    assert np.isclose(vals[0], 2.0 / 3.0)  # EOS (2 out of 3 in train)
    assert vals[1] == 0.0  # UNKNOWN_TOKEN
    assert vals[2] == 0.0  # ANOTHER_NEW_TOKEN


def test_validate_reports_mismatches_without_raising(
    synthetic_df: pd.DataFrame, sample_cfg: dict[str, Any]
) -> None:
    """Test that validate reports mismatches with paper_reference without raising."""
    X, y, meta = split_columns(synthetic_df, sample_cfg)
    report = validate(synthetic_df, X, y, meta, sample_cfg)

    assert isinstance(report, dict)
    assert report["rows"] == 5
    assert report["class_counts"] == {0: 3, 1: 2}
    assert report["feature_count"] == 4
    assert report["total_columns"] == 8

    # Paper comparison checks
    comp = report["paper_comparison"]
    assert comp["rows"]["paper"] == 9841
    assert comp["rows"]["actual"] == 5
    assert comp["rows"]["matches"] is False

    assert comp["benign"]["paper"] == 7663
    assert comp["benign"]["actual"] == 3
    assert comp["benign"]["matches"] is False

    assert comp["suspicious"]["paper"] == 2178
    assert comp["suspicious"]["actual"] == 2
    assert comp["suspicious"]["matches"] is False


@pytest.mark.skipif(
    not Path("data/transaction_dataset.csv").exists(),
    reason="Raw dataset data/transaction_dataset.csv not found",
)
def test_real_data_pipeline_execution(sample_cfg: dict[str, Any]) -> None:
    """Integration test: run full load, split, validate, encode pipeline on real dataset."""
    df, mapping = load_raw(sample_cfg)
    assert len(df) == 9841
    assert len(mapping) == 51

    X, y, meta = split_columns(df, sample_cfg)
    assert X.shape == (9841, 47)
    assert len(y) == 9841
    assert meta.shape == (9841, 3)

    report = validate(df, X, y, meta, sample_cfg)
    assert report["rows"] == 9841
    assert report["class_counts"] == {0: 7662, 1: 2179}
    assert report["paper_comparison"]["rows"]["matches"] is True
    assert report["paper_comparison"]["benign"]["matches"] is False
    assert report["paper_comparison"]["suspicious"]["matches"] is False

    encoder = TokenFrequencyEncoder(columns=sample_cfg["data"]["categorical_columns"])
    X_enc = encoder.fit_transform(X)
    assert X_enc.shape == (9841, 47)
    assert not X_enc["erc20_most_sent_token_type"].isna().any()
    assert not X_enc["erc20_most_rec_token_type"].isna().any()
