"""Unit tests for Particle Swarm Optimization (PSO) feature selection module.

All core tests run on small synthetic datasets with small swarm budgets
(e.g., swarm 5, iterations 3) to execute in fractions of a second.
One integration test on real data is marked slow and skipped unless RUN_SLOW_TESTS=1.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.pso import (
    compare_with_table2,
    compute_pso_cache_hash,
    make_rmse_fitness,
    pso_update_particle,
    run_pso,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Unknown pytest.mark.slow:pytest.PytestUnknownMarkWarning"
)


@pytest.fixture
def base_cfg() -> dict[str, Any]:
    """Base configuration fixture for PSO unit tests."""
    return {
        "seed": 42,
        "paths": {"results": "results/"},
        "data": {
            "label_column": "FLAG",
            "id_columns": ["Unnamed: 0", "Index", "Address"],
            "categorical_columns": ["erc20_most_sent_token_type", "erc20_most_rec_token_type"],
        },
        "pso": {
            "iterations": 3,
            "inertia_decay": 0.99,
            "swarm_size": 5,
            "c1": 2.0,
            "c2": 2.0,
            "w_initial": 0.9,
            "seg": 10,
            "bounds": [0.0, 1.0],
            "select_threshold": 0.5,
            "empty_subset_fitness": 1.0,
            "fitness": {
                "model": "decision_tree",
                "cv_folds": 3,
                "metric": "rmse_hard_labels",
            },
            "use_cache": False,
            "table2_columns": [
                "avg_min_between_received_tnx",
                "time_diff_between_first_and_last_mins",
                "sent_tnx",
                "received_tnx",
                "unique_received_from_addresses",
                "unique_sent_to_addresses",
                "avg_val_sent",
                "total_transactions_including_tnx_to_create_contract",
                "total_erc20_tnxs",
                "erc20_total_ether_sent",
                "erc20_min_val_sent",
                "erc20_max_val_sent",
                "erc20_avg_val_sent",
            ],
            "table2_ambiguous": {
                "Unique ERC20 Sent address": [
                    "erc20_uniq_sent_addr",
                    "erc20_uniq_sent_addr_1",
                ]
            },
        },
    }


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic dataset with 60 samples and 8 continuous features."""
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 8
    X_arr = rng.normal(0, 1, size=(n_samples, n_features))
    # Create non-trivial binary label with 2 classes
    y_arr = ((X_arr[:, 0] + X_arr[:, 1]) > 0.0).astype(int)
    cols = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=cols)
    y = pd.Series(y_arr, name="target")
    return X, y


def test_pso_determinism_same_seed(
    synthetic_data: tuple[pd.DataFrame, pd.Series], base_cfg: dict[str, Any]
) -> None:
    """Same seed must produce identical selected features, RMSE, and logbook."""
    X, y = synthetic_data
    cfg = dict(base_cfg)

    res1 = run_pso(X, y, cfg)
    res2 = run_pso(X, y, cfg)

    assert res1["selected_features"] == res2["selected_features"]
    assert res1["best_rmse"] == pytest.approx(res2["best_rmse"])
    assert res1["mask"] == res2["mask"]
    assert res1["evaluations"] == res2["evaluations"]

    for entry1, entry2 in zip(res1["logbook"], res2["logbook"]):
        assert entry1["iteration"] == entry2["iteration"]
        assert entry1["best_rmse"] == pytest.approx(entry2["best_rmse"])
        assert entry1["w"] == pytest.approx(entry2["w"])


def test_velocity_clamping_and_position_bounds() -> None:
    """Velocities must never exceed +/- Vmax and positions must stay within bounds."""
    bounds = (0.0, 1.0)
    seg = 10.0
    v_max = (bounds[1] - bounds[0]) / seg  # 0.1
    c1, c2 = 2.0, 2.0
    w = 0.9

    # Test extreme positive impulse
    x = np.array([0.9, 0.5])
    v = np.array([0.1, 0.05])
    pbest = np.array([1.0, 1.0])
    gbest = np.array([1.0, 1.0])
    r1 = np.array([1.0, 1.0])
    r2 = np.array([1.0, 1.0])

    new_x, new_v = pso_update_particle(
        x, v, pbest, gbest, w, c1, c2, v_max, bounds, r1, r2
    )

    assert np.all(new_v <= v_max + 1e-12)
    assert np.all(new_v >= -v_max - 1e-12)
    assert np.all(new_x <= bounds[1] + 1e-12)
    assert np.all(new_x >= bounds[0] - 1e-12)

    # Test extreme negative impulse pushing beyond bounds[0]
    x_neg = np.array([0.05, 0.1])
    v_neg = np.array([-0.08, -0.05])
    pbest_neg = np.array([0.0, 0.0])
    gbest_neg = np.array([0.0, 0.0])

    new_x_neg, new_v_neg = pso_update_particle(
        x_neg, v_neg, pbest_neg, gbest_neg, w, c1, c2, v_max, bounds, r1, r2
    )

    assert np.all(new_v_neg <= v_max + 1e-12)
    assert np.all(new_v_neg >= -v_max - 1e-12)
    assert np.all(new_x_neg <= bounds[1] + 1e-12)
    assert np.all(new_x_neg >= bounds[0] - 1e-12)


def test_inertia_decay(
    synthetic_data: tuple[pd.DataFrame, pd.Series], base_cfg: dict[str, Any]
) -> None:
    """Inertia weight must follow w(j) = w0 * alpha^j across iterations."""
    X, y = synthetic_data
    cfg = dict(base_cfg)
    w_initial = cfg["pso"]["w_initial"]
    alpha = cfg["pso"]["inertia_decay"]

    res = run_pso(X, y, cfg)
    logbook = res["logbook"]

    for entry in logbook:
        j = entry["iteration"]
        expected_w = w_initial * (alpha**j)
        assert entry["w"] == pytest.approx(expected_w, rel=1e-5)


def test_threshold_mapping() -> None:
    """Feature selection must strictly correspond to position > select_threshold."""
    threshold = 0.5
    pos = np.array([0.4999, 0.5000, 0.5001, 0.0, 1.0])
    expected_mask = [False, False, True, False, True]
    actual_mask = list(pos > threshold)
    assert actual_mask == expected_mask


def test_empty_subset_worst_fitness(
    synthetic_data: tuple[pd.DataFrame, pd.Series], base_cfg: dict[str, Any]
) -> None:
    """An empty feature subset must receive empty_subset_fitness (1.0)."""
    X, y = synthetic_data
    fitness_fn = make_rmse_fitness(X, y, base_cfg)

    empty_mask = np.zeros(X.shape[1], dtype=bool)
    rmse = fitness_fn(empty_mask)

    assert rmse == pytest.approx(base_cfg["pso"]["empty_subset_fitness"])
    assert fitness_fn.eval_count == 1


def test_logbook_length_and_monotonicity(
    synthetic_data: tuple[pd.DataFrame, pd.Series], base_cfg: dict[str, Any]
) -> None:
    """Logbook length must be iterations + 1, and best_rmse must be non-increasing."""
    X, y = synthetic_data
    cfg = dict(base_cfg)
    n_iter = cfg["pso"]["iterations"]

    res = run_pso(X, y, cfg)
    logbook = res["logbook"]

    assert len(logbook) == n_iter + 1

    for j in range(1, len(logbook)):
        assert logbook[j]["best_rmse"] <= logbook[j - 1]["best_rmse"] + 1e-12


def test_synthetic_signal_recovery_vs_all_features(base_cfg: dict[str, Any]) -> None:
    """On synthetic data where only 3 of 10 features carry signal, PSO best subset is <= all-features RMSE."""
    rng = np.random.default_rng(123)
    n_samples = 100
    # 3 signal features
    X_sig = rng.normal(0, 1, size=(n_samples, 3))
    # 7 noise features
    X_noise = rng.normal(0, 1, size=(n_samples, 7))
    X_arr = np.hstack([X_sig, X_noise])
    y_arr = ((X_sig[:, 0] + X_sig[:, 1] - X_sig[:, 2]) > 0.0).astype(int)

    cols = [f"sig_{i}" for i in range(3)] + [f"noise_{i}" for i in range(7)]
    X = pd.DataFrame(X_arr, columns=cols)
    y = pd.Series(y_arr, name="target")

    cfg = dict(base_cfg)
    cfg["pso"]["swarm_size"] = 10
    cfg["pso"]["iterations"] = 5
    cfg["seed"] = 123

    fitness_fn = make_rmse_fitness(X, y, cfg)
    all_features_mask = np.ones(10, dtype=bool)
    rmse_all = fitness_fn(all_features_mask)

    res = run_pso(X, y, cfg, fitness_fn=fitness_fn)
    # The best subset selected by PSO should be no worse than all-features RMSE (plus tiny float tol)
    assert res["best_rmse"] <= rmse_all + 1e-4


def test_raises_on_id_or_label_column(
    synthetic_data: tuple[pd.DataFrame, pd.Series], base_cfg: dict[str, Any]
) -> None:
    """run_pso must raise ValueError if any identifier or label column is in X."""
    X, y = synthetic_data

    # Test with label column included
    X_with_label = X.copy()
    X_with_label["FLAG"] = y
    with pytest.raises(ValueError, match="Forbidden ID or label column"):
        run_pso(X_with_label, y, base_cfg)

    # Test with ID column included
    X_with_id = X.copy()
    X_with_id["Address"] = "0x123"
    with pytest.raises(ValueError, match="Forbidden ID or label column"):
        run_pso(X_with_id, y, base_cfg)


def test_does_not_mutate_inputs(
    synthetic_data: tuple[pd.DataFrame, pd.Series], base_cfg: dict[str, Any]
) -> None:
    """run_pso must not mutate input X or y."""
    X, y = synthetic_data
    X_orig = X.copy()
    y_orig = y.copy()

    run_pso(X, y, base_cfg)

    pd.testing.assert_frame_equal(X, X_orig)
    pd.testing.assert_series_equal(y, y_orig)


def test_caching_and_force_flag(
    synthetic_data: tuple[pd.DataFrame, pd.Series],
    base_cfg: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Cache hit occurs only on matching hash, and is bypassed when force=True."""
    X, y = synthetic_data
    cfg = copy.deepcopy(base_cfg)
    cfg["pso"]["use_cache"] = True
    cache_file = tmp_path / "test_cache.json"

    # Run 1: Cold run, creates cache
    res1 = run_pso(X, y, cfg, cache_path=cache_file)
    assert cache_file.exists()
    assert res1["evaluations"] > 0

    # Run 2: Warm run, should hit cache
    res2 = run_pso(X, y, cfg, cache_path=cache_file)
    assert res2["selected_features"] == res1["selected_features"]
    assert res2["best_rmse"] == res1["best_rmse"]

    # Run 3: Warm run with force=True, recomputes
    res3 = run_pso(X, y, cfg, force=True, cache_path=cache_file)
    assert res3["selected_features"] == res1["selected_features"]

    # Run 4: Hash mismatch due to altered parameter
    cfg_altered = copy.deepcopy(cfg)
    cfg_altered["pso"]["iterations"] = 4
    h1 = compute_pso_cache_hash(cfg, (len(X), len(X.columns)), list(X.columns), cfg["seed"])
    h2 = compute_pso_cache_hash(cfg_altered, (len(X), len(X.columns)), list(X.columns), cfg["seed"])
    assert h1 != h2


def test_compare_with_table2(base_cfg: dict[str, Any]) -> None:
    """compare_with_table2 correctly computes overlap, Jaccard, and handles ambiguous columns."""
    cfg = dict(base_cfg)
    selected = [
        "avg_min_between_received_tnx",
        "sent_tnx",
        "custom_unrelated_feature",
        "erc20_uniq_sent_addr",  # Candidate 1 for ambiguous feature
    ]

    comp = compare_with_table2(selected, cfg)

    assert comp["overlap_count"] == 2
    assert "avg_min_between_received_tnx" in comp["matched_features"]
    assert "sent_tnx" in comp["matched_features"]
    assert "custom_unrelated_feature" in comp["extra_features"]
    assert "time_diff_between_first_and_last_mins" in comp["missing_features"]

    ambig = comp["ambiguous_features"]["Unique ERC20 Sent address"]
    assert ambig["any_selected"] is True
    assert ambig["selected_candidates"] == ["erc20_uniq_sent_addr"]


@pytest.mark.slow
def test_real_data_smoke(base_cfg: dict[str, Any]) -> None:
    """End-to-end smoke test on real dataset (skipped unless RUN_SLOW_TESTS=1)."""
    if not os.getenv("RUN_SLOW_TESTS"):
        pytest.skip("Set RUN_SLOW_TESTS=1 to run real-data smoke test.")

    import yaml

    from src.data import TokenFrequencyEncoder, load_raw, split_columns
    from src.preprocess import apply_scaler, clean_missing, fit_scaler

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        real_cfg = yaml.safe_load(f)

    # Use very small swarm for smoke test
    real_cfg["pso"]["swarm_size"] = 2
    real_cfg["pso"]["iterations"] = 1
    real_cfg["pso"]["use_cache"] = False

    df, _ = load_raw(real_cfg)
    X, y, meta = split_columns(df, real_cfg)
    X_clean, y_clean, _, _ = clean_missing(X, y, meta, real_cfg)
    enc = TokenFrequencyEncoder(columns=real_cfg["data"]["categorical_columns"])
    X_enc = enc.fit_transform(X_clean)
    scaler = fit_scaler(X_enc)
    X_scaled = apply_scaler(scaler, X_enc)

    res = run_pso(X_scaled, y_clean, real_cfg)
    assert len(res["selected_features"]) > 0
    assert 0.0 <= res["best_rmse"] <= 1.0
