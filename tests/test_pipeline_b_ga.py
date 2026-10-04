"""Tests for Task 2.0c: Duplicate Diagnostics and GA on the Ladder.

Verifies:
1. compute_duplicate_diagnostics correctly identifies duplicate rows, contradictory groups,
   and train/test feature overlap on a synthetic hand-built DataFrame.
2. Rung B GA calls run_ga with pipeline="pipeline_b" and train_index_hash, without writing or
   modifying pipeline_a caches.
3. Rung L1 GA strictly rejects non-xgboost models (e.g. svm, isolation_forest).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.diagnostics import compute_duplicate_diagnostics
from src.pipeline_b import run_pipeline_b_ga


@pytest.fixture
def base_cfg() -> dict[str, Any]:
    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_diagnostics_counts_on_hand_built_frame() -> None:
    """Test duplicate and contradiction diagnostics on a hand-built DataFrame with known properties."""
    # Build a small DataFrame with known duplicate feature rows and contradictory labels:
    # Row 0: [1.0, 2.0], label 0
    # Row 1: [1.0, 2.0], label 1  <- CONTRADICTORY with Row 0 (same features, different labels)
    # Row 2: [3.0, 4.0], label 0
    # Row 3: [3.0, 4.0], label 0  <- DUPLICATE with Row 2 (same features, same label)
    # Row 4: [5.0, 6.0], label 1  <- UNIQUE
    X_all = pd.DataFrame(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        columns=["f1", "f2"],
        index=[0, 1, 2, 3, 4],
    )
    y_all = pd.Series([0, 1, 0, 0, 1], index=[0, 1, 2, 3, 4])

    # Partition into Train: rows 0, 2, 4 and Test: rows 1, 3
    # Note:
    # Test row 1 ([1.0, 2.0]) appears in train (row 0 has [1.0, 2.0])
    # Test row 3 ([3.0, 4.0]) appears in train (row 2 has [3.0, 4.0])
    # So 2 test rows appear in train, 0 unseen test rows
    X_train = X_all.loc[[0, 2, 4]]
    y_train = y_all.loc[[0, 2, 4]]
    X_test = X_all.loc[[1, 3]]
    y_test = y_all.loc[[1, 3]]

    res = compute_duplicate_diagnostics(X_all, y_all, X_train, y_train, X_test, y_test)

    dups = res["duplicate_rows"]
    contra = res["contradictory_groups"]
    overlap = res["train_test_overlap"]

    # Redundant rows (keep='first'): rows 1 and 3 -> count 2
    assert dups["redundant_rows_count"] == 2
    # Cluster rows (keep=False): rows 0, 1, 2, 3 -> count 4
    assert dups["cluster_rows_count"] == 4

    # Contradictory groups: exactly 1 group ([1.0, 2.0] has labels [0, 1])
    assert contra["groups_count"] == 1
    assert contra["rows_count"] == 2

    # Overlap: both test rows appear in train
    assert overlap["test_rows_in_train_count"] == 2
    assert overlap["unseen_test_count"] == 0

    # Test with an unseen test row
    X_test_unseen = pd.DataFrame([[99.0, 99.0]], columns=["f1", "f2"], index=[5])
    y_test_unseen = pd.Series([0], index=[5])
    res_unseen = compute_duplicate_diagnostics(
        X_all, y_all, X_train, y_train, X_test_unseen, y_test_unseen
    )
    assert res_unseen["train_test_overlap"]["unseen_test_count"] == 1
    assert res_unseen["train_test_overlap"]["test_rows_in_train_count"] == 0


def test_rung_l1_ga_rejects_non_xgboost(base_cfg: dict[str, Any]) -> None:
    """Assert Rung L1 GA strictly rejects non-xgboost models."""
    with pytest.raises(ValueError, match="Rung L1 GA only supports xgboost"):
        run_pipeline_b_ga(model_name="svm", rung="l1", cfg=base_cfg)

    with pytest.raises(ValueError, match="Rung L1 GA only supports xgboost"):
        run_pipeline_b_ga(model_name="isolation_forest", rung="l1", cfg=base_cfg)


def test_rung_b_ga_calls_run_ga_with_pipeline_b_and_train_index_hash(
    base_cfg: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Assert Rung B GA calls run_ga with pipeline='pipeline_b' and preserves pipeline_a caches."""
    # Record hashes of existing Pipeline A caches if they exist
    results_dir = Path(base_cfg.get("paths", {}).get("results", "results/"))
    pso_a_cache = results_dir / "cache" / "pso_pipeline_a.json"
    pso_a_hash = None
    if pso_a_cache.exists():
        pso_a_hash = hashlib.sha256(pso_a_cache.read_bytes()).hexdigest()

    ga_a_caches = list((results_dir / "cache").glob("ga_pipeline_a_*.json"))
    ga_a_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in ga_a_caches}

    # Mock run_ga to inspect arguments and avoid running real 20-gen GA in unit test
    mock_ga_result = {
        "model_name": "xgboost",
        "feature_set": "pso",
        "best_params": {
            "n_estimators": 50,
            "gamma": 0.1,
            "min_child_weight": 2,
            "colsample_bytree": 0.8,
            "max_depth": 4,
            "reg_lambda": 1.0,
            "learning_rate": 0.1,
        },
        "best_fitness": 0.95,
        "best_individual": [50.0, 0.1, 2.0, 0.8, 4.0, 1.0, 0.1],
        "evaluations": 10,
        "logbook": [
            {"gen": 0, "nevals": 10, "avg": 0.9, "std": 0.05, "min": 0.8, "max": 0.95}
        ],
        "elapsed_seconds": 1.2,
        "seed": 42,
        "cached": False,
    }

    with patch("src.pipeline_b.run_ga", return_value=mock_ga_result) as mock_run_ga:
        res = run_pipeline_b_ga(
            model_name="xgboost",
            rung="b",
            cfg=base_cfg,
            force=False,
        )

        assert mock_run_ga.called
        call_kwargs = mock_run_ga.call_args.kwargs
        assert call_kwargs.get("pipeline") == "pipeline_b"
        assert call_kwargs.get("train_index_hash") is not None
        assert isinstance(call_kwargs.get("train_index_hash"), str)
        assert len(call_kwargs.get("train_index_hash")) == 64  # SHA-256 length

    # Assert Pipeline A caches are completely untouched
    if pso_a_hash is not None:
        assert hashlib.sha256(pso_a_cache.read_bytes()).hexdigest() == pso_a_hash

    for p, original_hash in ga_a_hashes.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == original_hash
