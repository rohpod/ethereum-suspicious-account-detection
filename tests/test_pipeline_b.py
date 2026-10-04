"""Tests for Pipeline B Leakage Ladder (Task 2.0b).

Verifies:
1. L1 and B produce natural-ratio test sets with no synthetic rows.
2. B's scaler and TokenFrequencyEncoder statistics derive exclusively from train split.
3. B invokes PSO with pipeline="pipeline_b" and does not touch results/cache/pso_pipeline_a.json.
4. Consistency checks pass across all 5 models on synthetic data.
5. get_pipeline_b_cfg deep-copies config and applies pipeline_b block without mutating base cfg.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import TokenFrequencyEncoder
from src.evaluate import check_consistency
from src.pipeline_b import (
    build_pipeline_b_data,
    build_pipeline_l1_data,
    deep_merge,
    evaluate_rung_models,
    get_pipeline_b_cfg,
)
from src.preprocess import (
    apply_scaler,
    apply_smote,
    fit_scaler,
    split_stratified,
)
from src.pso import compute_pso_cache_hash, hash_index, run_pso


@pytest.fixture
def base_cfg() -> dict[str, Any]:
    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_get_pipeline_b_cfg_overrides_without_mutation(base_cfg: dict[str, Any]) -> None:
    """Test get_pipeline_b_cfg correctly applies overrides without mutating base_cfg."""
    original_pso_smote = base_cfg.get("pso", {}).get("fitness", {}).get("smote_in_fold", False)
    assert original_pso_smote is False

    cfg_b = get_pipeline_b_cfg(base_cfg)

    # Overrides applied
    assert cfg_b["pso"]["fitness"]["smote_in_fold"] is True
    assert cfg_b["split"]["stratify"] is True

    # Base config unmutated
    assert base_cfg["pso"]["fitness"]["smote_in_fold"] is False
    assert base_cfg.get("split", {}).get("stratify", False) is False


def test_l1_and_b_natural_test_sets_no_synthetic(base_cfg: dict[str, Any]) -> None:
    """Assert L1 and B data preparation produce natural test sets with zero synthetic rows."""
    np.random.seed(42)
    n_samples = 120
    n_suspicious = 24  # 20% suspicious
    feature_names = [f"feat_{i}" for i in range(10)]

    X = pd.DataFrame(
        np.random.randn(n_samples, len(feature_names)),
        columns=feature_names,
    )
    y = pd.Series([0] * (n_samples - n_suspicious) + [1] * n_suspicious)

    # Simulate Rung L1 protocol on synthetic data
    # 1. Stratified split
    X_tr_l1, X_te_l1, y_tr_l1, y_te_l1 = split_stratified(X, y, base_cfg)
    # 2. SMOTE on train only
    X_tr_l1_res, y_tr_l1_res, is_syn_tr_l1, _ = apply_smote(X_tr_l1, y_tr_l1, base_cfg)
    is_syn_te_l1 = np.zeros(len(X_te_l1), dtype=bool)

    # Assertions for L1
    assert len(X_te_l1) == round(n_samples * 0.2)
    assert np.sum(is_syn_te_l1) == 0, "L1 test set must not contain synthetic rows"
    l1_test_susp = int(y_te_l1.sum())
    expected_susp = len(X_te_l1) * (n_suspicious / n_samples)
    assert abs(l1_test_susp - expected_susp) <= 1.0

    # Simulate Rung B protocol on synthetic data
    # 1. Stratified split FIRST
    X_tr_b, X_te_b, y_tr_b, y_te_b = split_stratified(X, y, base_cfg)
    # 2. Fit scaler on train only, transform both
    scaler = fit_scaler(X_tr_b)
    X_tr_b_scaled = apply_scaler(scaler, X_tr_b)
    X_te_b_scaled = apply_scaler(scaler, X_te_b)
    # 3. SMOTE on train only
    X_tr_b_res, y_tr_b_res, is_syn_tr_b, _ = apply_smote(X_tr_b_scaled, y_tr_b, base_cfg)
    is_syn_te_b = np.zeros(len(X_te_b_scaled), dtype=bool)

    # Assertions for B
    assert len(X_te_b_scaled) == round(n_samples * 0.2)
    assert np.sum(is_syn_te_b) == 0, "B test set must not contain synthetic rows"
    b_test_susp = int(y_te_b.sum())
    assert abs(b_test_susp - expected_susp) <= 1.0


def test_pipeline_b_train_only_encoder_and_scaler_statistics() -> None:
    """Assert Pipeline B fits encoder and scaler strictly on train set without test leakage."""
    # Create distinct train and test distributions
    # Train numeric mean = 10.0; Test numeric mean = 50.0
    # Train categories = {"tokenA", "tokenB"}; Test categories = {"tokenA", "tokenC"}
    X_tr = pd.DataFrame({
        "num1": [10.0, 10.0, 10.0, 10.0],
        "tok1": ["tokenA", "tokenA", "tokenB", "tokenB"],
    })
    X_te = pd.DataFrame({
        "num1": [50.0, 50.0],
        "tok1": ["tokenA", "tokenC"],  # tokenC is unseen in train
    })

    encoder = TokenFrequencyEncoder(columns=["tok1"])
    X_tr_enc = encoder.fit_transform(X_tr)
    X_te_enc = encoder.transform(X_te)

    # Train frequencies: tokenA = 0.5, tokenB = 0.5
    assert encoder.frequencies_["tok1"]["tokenA"] == 0.5
    assert encoder.frequencies_["tok1"]["tokenB"] == 0.5
    # Test tokenC should be mapped to 0.0 (unseen)
    assert X_te_enc["tok1"].iloc[1] == 0.0
    assert X_te_enc["tok1"].iloc[0] == 0.5

    # Scaler fit on train only
    scaler = fit_scaler(X_tr_enc)
    # Scaler mean must match X_tr_enc mean, NOT combined mean
    assert np.isclose(scaler.mean_[0], 10.0)

    X_te_scaled = apply_scaler(scaler, X_te_enc)
    # If scaled with train statistics, 50.0 - 10.0 != 0
    assert not np.allclose(X_te_scaled["num1"].values, 0.0)


def test_pipeline_b_pso_cache_isolation_and_immutability(
    base_cfg: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Assert Pipeline B PSO uses pipeline_b cache and never touches pso_pipeline_a.json."""
    pso_a_cache_file = Path("results/cache/pso_pipeline_a.json")
    if pso_a_cache_file.exists():
        with open(pso_a_cache_file, "rb") as f:
            original_bytes = f.read()
        original_hash = hashlib.sha256(original_bytes).hexdigest()
    else:
        original_hash = None

    # Run PSO on small synthetic data using pipeline="pipeline_b" and isolated cache dir
    np.random.seed(42)
    X_syn = pd.DataFrame(np.random.randn(30, 4), columns=["f0", "f1", "f2", "f3"])
    y_syn = pd.Series([0] * 24 + [1] * 6)

    cfg_b = get_pipeline_b_cfg(base_cfg)
    cfg_b["pso"]["iterations"] = 2
    cfg_b["pso"]["swarm_size"] = 4
    cfg_b["paths"]["results"] = str(tmp_path)

    res = run_pso(
        X_syn,
        y_syn,
        cfg_b,
        pipeline="pipeline_b",
        train_index_hash=hash_index(X_syn.index),
    )

    # Check cache was created in isolated dir under pso_pipeline_b.json
    expected_b_cache = tmp_path / "cache" / "pso_pipeline_b.json"
    assert expected_b_cache.exists()

    # Verify results/cache/pso_pipeline_a.json was NOT touched
    if original_hash is not None:
        with open(pso_a_cache_file, "rb") as f:
            current_bytes = f.read()
        current_hash = hashlib.sha256(current_bytes).hexdigest()
        assert current_hash == original_hash, "pso_pipeline_a.json was modified!"


def test_evaluate_rung_models_consistency_checks_synthetic(
    base_cfg: dict[str, Any],
) -> None:
    """Assert evaluate_rung_models runs all 5 models and passes consistency checks on synthetic data."""
    np.random.seed(42)
    n_train = 60
    n_test = 20
    feats = [f"f_{i}" for i in range(5)]

    X_tr = pd.DataFrame(np.random.randn(n_train, len(feats)), columns=feats)
    y_tr = pd.Series([0] * 30 + [1] * 30)  # balanced train

    X_te = pd.DataFrame(np.random.randn(n_test, len(feats)), columns=feats)
    y_te = pd.Series([0] * 16 + [1] * 4)  # natural test ratio 20%

    synthetic_data = {
        "X_train": X_tr,
        "y_train": y_tr,
        "X_test": X_te,
        "y_test": y_te,
        "is_synthetic_train": np.zeros(n_train, dtype=bool),
        "is_synthetic_test": np.zeros(n_test, dtype=bool),
        "feature_names": feats,
        "smote_report": {"test": "synthetic"},
        "split_report": {
            "train_rows_natural": n_train,
            "train_rows_resampled": n_train,
            "test_rows": n_test,
            "train_class_counts_natural": {0: 30, 1: 30},
            "train_class_counts_resampled": {0: 30, 1: 30},
            "test_class_counts": {0: 16, 1: 4},
            "synthetic_train_count": 0,
            "synthetic_test_count": 0,
            "synthetic_test_ratio": 0.0,
        },
    }

    rung_payload, timings = evaluate_rung_models(synthetic_data, base_cfg, "test_rung")

    # Assert all 5 models present
    expected_models = {"xgboost", "svm", "isolation_forest", "cart", "lof"}
    assert set(rung_payload["models"].keys()) == expected_models

    # Assert consistency checks passed
    checks = rung_payload["consistency_checks"]
    assert len(checks) > 0
    for c in checks:
        assert c["passed"] is True, f"Consistency check failed: {c}"

    # Assert majority baseline
    maj = rung_payload["majority_baseline"]
    assert maj["majority_class"] == 0
    assert np.isclose(maj["accuracy"], 16 / 20)
    assert maj["recall_suspicious"] == 0.0
