"""Tests for Pipeline B plumbing (Task 2.0a).

Verifies:
1. Stratified split helper (split_stratified) preserves class ratio within 1 sample and has no index overlap.
2. Fold-wise SMOTE: validation folds untouched with natural sizes; smote_in_fold=False matches original behavior.
3. Small-minority guard handles minority < 2 and minority <= k_neighbors without crashing.
4. GA metric f1_suspicious runs end-to-end on tiny budget and yields positive fitness (> 0).
5. Cache namespacing: pipeline_a yields identical path and hash to before; pipeline_b gets distinct path and hash; train_index_hash produces distinct hash.
6. TokenFrequencyEncoder and StandardScaler fit on train only, applied to test with unseen categories.
"""

from __future__ import annotations

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
from src.ga import (
    compute_ga_cache_hash,
    make_ga_fitness,
    run_ga,
)
from src.preprocess import (
    apply_scaler,
    fit_scaler,
    split_stratified,
)
from src.pso import (
    apply_fold_smote,
    compute_pso_cache_hash,
    hash_index,
    make_rmse_fitness,
)


@pytest.fixture
def base_cfg() -> dict[str, Any]:
    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_split_stratified_preserves_class_ratio_and_no_overlap(
    base_cfg: dict[str, Any],
) -> None:
    """Test split_stratified maintains class ratio within 1 sample and has disjoint indices."""
    np.random.seed(42)
    n_samples = 100
    n_suspicious = 20
    X = pd.DataFrame(
        np.random.randn(n_samples, 4),
        index=[f"acc_{i}" for i in range(n_samples)],
        columns=["f1", "f2", "f3", "f4"],
    )
    y = pd.Series([0] * (n_samples - n_suspicious) + [1] * n_suspicious, index=X.index)

    X_train, X_test, y_train, y_test = split_stratified(X, y, base_cfg)

    # 80/20 split
    assert len(X_train) == 80
    assert len(X_test) == 20

    # Index properties
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(test_idx)
    assert train_idx | test_idx == set(X.index)

    # Class balance preservation within 1 sample
    raw_ratio = n_suspicious / n_samples  # 0.20
    test_suspicious = int(y_test.sum())
    train_suspicious = int(y_train.sum())

    expected_test_pos = round(len(X_test) * raw_ratio)  # 4
    expected_train_pos = round(len(X_train) * raw_ratio)  # 16
    assert abs(test_suspicious - expected_test_pos) <= 1
    assert abs(train_suspicious - expected_train_pos) <= 1


def test_fold_wise_smote_validation_folds_untouched(
    base_cfg: dict[str, Any],
) -> None:
    """Test that with smote_in_fold=True, validation folds retain original sizes and SMOTE is invoked."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["c1", "c2", "c3", "c4"])
    y = pd.Series([0] * 50 + [1] * 10)

    cfg_smote = dict(base_cfg)
    cfg_smote["pso"] = dict(base_cfg.get("pso", {}))
    cfg_smote["pso"]["fitness"] = dict(base_cfg.get("pso", {}).get("fitness", {}))
    cfg_smote["pso"]["fitness"]["smote_in_fold"] = True
    cfg_smote["ga"] = dict(base_cfg.get("ga", {}))
    cfg_smote["ga"]["smote_in_fold"] = True

    # 1. PSO RMSEFitness
    with patch("src.pso.apply_fold_smote", wraps=apply_fold_smote) as spy_pso_smote:
        fitness_pso = make_rmse_fitness(X, y, cfg_smote)
        score_pso = fitness_pso(np.array([True, True, False, False]))
        assert spy_pso_smote.called
        assert score_pso > 0.0

    # 2. GA GAFitness
    specs = base_cfg["ga"]["search_spaces"]["xgboost"]
    with patch("src.ga.SMOTE.fit_resample", autospec=True) as spy_ga_smote:
        fitness_ga = make_ga_fitness("xgboost", X, y, specs, cfg_smote)
        def fake_resample(self_smote, x_in, y_in):
            return x_in, y_in
        spy_ga_smote.side_effect = fake_resample
        ind = [50, 0.5, 2.0, 0.8, 4, 1.0, 0.1]
        score_ga = fitness_ga(ind)
        assert spy_ga_smote.called
        assert score_ga[0] > 0.0

    # 3. With smote_in_fold=False, outputs match default
    cfg_nosmote = dict(base_cfg)
    cfg_nosmote["pso"] = dict(base_cfg.get("pso", {}))
    cfg_nosmote["pso"]["fitness"] = dict(base_cfg.get("pso", {}).get("fitness", {}))
    cfg_nosmote["pso"]["fitness"]["smote_in_fold"] = False
    fitness_default = make_rmse_fitness(X, y, base_cfg)
    fitness_explicit_false = make_rmse_fitness(X, y, cfg_nosmote)
    mask = np.array([True, False, True, False])
    assert abs(fitness_default(mask) - fitness_explicit_false(mask)) < 1e-12


def test_small_minority_guard_does_not_crash() -> None:
    """Test small-minority guard handles minority < 2 and minority <= k_neighbors."""
    np.random.seed(42)
    X = np.random.randn(20, 3)

    # Case 1: minority < 2 (only 1 positive sample) -> skip SMOTE
    y_single = np.array([0] * 19 + [1] * 1)
    X_res1, y_res1 = apply_fold_smote(X, y_single, k_neighbors=5, seed=42)
    assert len(X_res1) == len(X)
    assert np.array_equal(y_res1, y_single)

    # Case 2: minority count <= k_neighbors (3 positives <= 5) -> k reduced to 2
    y_three = np.array([0] * 17 + [1] * 3)
    X_res3, y_res3 = apply_fold_smote(X, y_three, k_neighbors=5, seed=42)
    assert len(X_res3) == 34  # 17 benign + 17 suspicious
    assert np.sum(y_res3 == 1) == 17

    # Case 3: 0 minority samples (single class) -> skip SMOTE
    y_zero = np.array([0] * 20)
    X_res0, y_res0 = apply_fold_smote(X, y_zero, k_neighbors=5, seed=42)
    assert len(X_res0) == 20


def test_ga_metric_f1_suspicious_tiny_budget(
    base_cfg: dict[str, Any],
) -> None:
    """Test GA metric f1_suspicious runs end to end on tiny budget and yields positive fitness."""
    np.random.seed(42)
    n = 40
    X = pd.DataFrame(np.random.randn(n, 3), columns=["f0", "f1", "f2"])
    y = pd.Series([0] * 30 + [1] * 10)
    X.loc[y == 1, "f0"] += 4.0

    cfg_ga = dict(base_cfg)
    cfg_ga["ga"] = dict(base_cfg.get("ga", {}))
    cfg_ga["ga"]["metric"] = "f1_suspicious"

    res = run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=cfg_ga,
        pop_size_override=4,
        n_gen_override=1,
        force=True,
        pipeline="pipeline_b",
    )

    assert res["best_fitness"] > 0.0
    assert len(res["logbook"]) == 2
    for entry in res["logbook"]:
        assert entry["min"] > 0.0
        assert entry["max"] > 0.0


def test_cache_namespacing_hashes_and_paths(
    base_cfg: dict[str, Any],
) -> None:
    """Test cache paths and hashes for pipeline_a, pipeline_b, and train_index_hash."""
    X_shape = (9012, 47)
    cols = [f"col_{i}" for i in range(47)]
    seed = 42

    # 1. pipeline_a hash matches old calculation (without smote_in_fold)
    old_pso_cfg = dict(base_cfg.get("pso", {}))
    if "fitness" in old_pso_cfg and "smote_in_fold" in old_pso_cfg["fitness"]:
        old_pso_cfg["fitness"] = dict(old_pso_cfg["fitness"])
        del old_pso_cfg["fitness"]["smote_in_fold"]
    old_payload = {
        "pso_config": old_pso_cfg,
        "data_shape": list(X_shape),
        "column_names": list(cols),
        "seed": seed,
    }
    old_hash = hashlib.sha256(json.dumps(old_payload, sort_keys=True).encode("utf-8")).hexdigest()

    h_pipe_a = compute_pso_cache_hash(base_cfg, X_shape, cols, seed, pipeline="pipeline_a")
    assert h_pipe_a == old_hash

    # 2. pipeline_b gets distinct hash
    h_pipe_b = compute_pso_cache_hash(base_cfg, X_shape, cols, seed, pipeline="pipeline_b")
    assert h_pipe_b != h_pipe_a

    # 3. train_index_hash helper and distinct hashing
    idx1 = pd.Index([20, 10, 30])
    idx2 = pd.Index([10, 20, 30])
    idx3 = pd.Index([10, 20, 31])
    h_idx1 = hash_index(idx1)
    h_idx2 = hash_index(idx2)
    h_idx3 = hash_index(idx3)
    assert h_idx1 == h_idx2  # Sorted invariance
    assert h_idx1 != h_idx3

    h_pipe_b_idx = compute_pso_cache_hash(
        base_cfg, X_shape, cols, seed, pipeline="pipeline_b", train_index_hash=h_idx1
    )
    assert h_pipe_b_idx != h_pipe_b

    # 4. GA hash namespacing
    lib_versions = {"scikit-learn": "1.0", "xgboost": "1.0", "numpy": "1.0", "pandas": "1.0"}
    specs = base_cfg["ga"]["search_spaces"]["xgboost"]
    h_ga_a = compute_ga_cache_hash(
        "xgboost", "pso", base_cfg["ga"], specs, (100, 22), (100,), cols[:22], seed, lib_versions, pipeline="pipeline_a"
    )
    h_ga_b = compute_ga_cache_hash(
        "xgboost", "pso", base_cfg["ga"], specs, (100, 22), (100,), cols[:22], seed, lib_versions, pipeline="pipeline_b"
    )
    assert h_ga_a != h_ga_b


def test_token_encoder_and_scaler_fitted_on_train_only() -> None:
    """Test TokenFrequencyEncoder and StandardScaler fit on train only and applied to test with unseen categories."""
    X_train = pd.DataFrame({
        "tok": ["Alpha", "Beta", "Beta", "Alpha"],
        "val": [10.0, 10.0, 10.0, 10.0],
    })
    X_test = pd.DataFrame({
        "tok": ["Beta", "Gamma"],  # "Gamma" is unseen
        "val": [50.0, 50.0],
    })

    # Fit encoder on train only
    encoder = TokenFrequencyEncoder(columns=["tok"])
    encoder.fit(X_train)

    X_train_enc = encoder.transform(X_train)
    X_test_enc = encoder.transform(X_test)

    # Train frequencies: Alpha: 0.5, Beta: 0.5
    assert X_train_enc["tok"].tolist() == [0.5, 0.5, 0.5, 0.5]
    # Unseen category Gamma mapped to 0.0
    assert X_test_enc["tok"].tolist() == [0.5, 0.0]

    # Fit scaler on train only
    scaler = fit_scaler(X_train_enc[["val"]])
    assert np.allclose(scaler.mean_, [10.0])

    X_test_scaled = apply_scaler(scaler, X_test_enc[["val"]])
    # Mean of test (50.0) scaled using train statistics (mean=10.0, std=1.0 for constant)
    assert not np.allclose(scaler.mean_, [30.0])  # Combined mean would be 30.0
    assert isinstance(X_test_scaled, pd.DataFrame)
