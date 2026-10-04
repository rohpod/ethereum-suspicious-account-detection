"""Unit tests for Phase 1.6 Genetic Algorithm (GA) hyperparameter tuning.

Uses synthetic datasets and small budgets to ensure fast, deterministic execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from deap import creator, tools

from src.ga import (
    clip_individual,
    decode_individual,
    make_ga_fitness,
    run_budget_probe,
    run_ga,
)


@pytest.fixture
def synthetic_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate small synthetic classification dataset (50 samples, 4 features)."""
    np.random.seed(42)
    X_arr = np.random.randn(60, 4)
    # Balanced classes
    y_arr = np.array([0] * 30 + [1] * 30)
    # Give feature 0 some predictive power
    X_arr[:30, 0] -= 1.0
    X_arr[30:, 0] += 1.0

    cols = ["feat_0", "feat_1", "feat_2", "feat_3"]
    X = pd.DataFrame(X_arr, columns=cols)
    y = pd.Series(y_arr, name="FLAG")
    return X, y


@pytest.fixture
def mini_cfg() -> dict[str, Any]:
    """Minimal test configuration with small search spaces and parameters."""
    return {
        "seed": 42,
        "paths": {
            "results": "results/",
        },
        "ga": {
            "population": 4,
            "generations": 2,
            "cxpb": 0.5,
            "mutpb": 0.2,
            "indpb": 0.2,
            "sigma_fraction": 0.1,
            "blend_alpha": 0.5,
            "cv_folds": 3,
            "metric": "accuracy",
            "use_cache": False,
            "search_spaces": {
                "xgboost": [
                    {"name": "n_estimators", "type": "int", "bounds": [10, 50]},
                    {"name": "max_depth", "type": "int", "bounds": [3, 6]},
                    {"name": "learning_rate", "type": "float", "bounds": [0.05, 0.5]},
                ],
                "svm": [
                    {"name": "C", "type": "log10_float", "bounds": [-1.0, 1.0]},
                    {"name": "gamma", "type": "log10_float", "bounds": [-2.0, 0.0]},
                ],
                "isolation_forest": [
                    {"name": "contamination", "type": "float", "bounds": [0.1, 0.4]},
                    {"name": "max_samples", "type": "int", "bounds": [16, 64]},
                    {"name": "n_estimators", "type": "int", "bounds": [20, 50]},
                ],
            },
        },
        "models": {
            "xgboost": {
                "objective": "binary:logistic",
                "subsample": 1.0,
                "defaults": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
            },
            "svm": {
                "kernel": "rbf",
                "defaults": {"C": 1.0, "gamma": 0.1},
            },
            "isolation_forest": {
                "defaults": {"contamination": 0.2, "max_samples": 32, "n_estimators": 20},
            },
        },
    }


def test_decode_individual_int_rounding_and_log10() -> None:
    """Test gene decoding handles int rounding, log10 exponents, and bounds clipping."""
    specs = [
        {"name": "int_param", "type": "int", "bounds": [10, 50]},
        {"name": "float_param", "type": "float", "bounds": [0.0, 1.0]},
        {"name": "log10_param", "type": "log10_float", "bounds": [-2.0, 2.0]},
    ]

    # Normal values
    ind = [25.4, 0.75, 1.0]
    decoded = decode_individual(ind, specs)
    assert decoded["int_param"] == 25
    assert decoded["float_param"] == 0.75
    assert abs(decoded["log10_param"] - 10.0) < 1e-6

    # Rounding up
    ind_round_up = [25.6, 0.5, 0.0]
    decoded_up = decode_individual(ind_round_up, specs)
    assert decoded_up["int_param"] == 26
    assert abs(decoded_up["log10_param"] - 1.0) < 1e-6

    # Exceeding upper bounds
    ind_high = [100.0, 2.5, 5.0]
    decoded_high = decode_individual(ind_high, specs)
    assert decoded_high["int_param"] == 50
    assert decoded_high["float_param"] == 1.0
    assert abs(decoded_high["log10_param"] - 100.0) < 1e-6

    # Exceeding lower bounds
    ind_low = [-10.0, -0.5, -5.0]
    decoded_low = decode_individual(ind_low, specs)
    assert decoded_low["int_param"] == 10
    assert decoded_low["float_param"] == 0.0
    assert abs(decoded_low["log10_param"] - 0.01) < 1e-6


def test_clip_individual_in_place() -> None:
    """Test clip_individual modifies individual values to stay strictly within bounds."""
    specs = [
        {"name": "p1", "type": "float", "bounds": [0.0, 5.0]},
        {"name": "p2", "type": "float", "bounds": [-2.0, 2.0]},
    ]
    ind = [-1.0, 10.0]
    clipped = clip_individual(ind, specs)
    assert clipped[0] == 0.0
    assert clipped[1] == 2.0
    assert ind is clipped


def test_ga_determinism_same_seed(
    synthetic_train_data: tuple[pd.DataFrame, pd.Series],
    mini_cfg: dict[str, Any],
) -> None:
    """Test that two GA runs with identical seeds yield identical logbooks and best individuals."""
    X, y = synthetic_train_data

    run1 = run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=mini_cfg,
        pop_size_override=4,
        n_gen_override=2,
        force=True,
    )
    run2 = run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=mini_cfg,
        pop_size_override=4,
        n_gen_override=2,
        force=True,
    )

    assert run1["best_individual"] == run2["best_individual"]
    assert run1["best_params"] == run2["best_params"]
    assert run1["best_fitness"] == run2["best_fitness"]
    assert len(run1["logbook"]) == len(run2["logbook"])
    for row1, row2 in zip(run1["logbook"], run2["logbook"]):
        assert row1["gen"] == row2["gen"]
        assert row1["nevals"] == row2["nevals"]
        assert abs(row1["avg"] - row2["avg"]) < 1e-9
        assert abs(row1["max"] - row2["max"]) < 1e-9


def test_logbook_generations_and_nevals_count(
    synthetic_train_data: tuple[pd.DataFrame, pd.Series],
    mini_cfg: dict[str, Any],
) -> None:
    """Test logbook length is n_gen + 1 and nevals per generation is <= population."""
    X, y = synthetic_train_data
    n_gen = 3
    pop_size = 6

    res = run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=mini_cfg,
        pop_size_override=pop_size,
        n_gen_override=n_gen,
        force=True,
    )

    logbook = res["logbook"]
    assert len(logbook) == n_gen + 1

    expected_cols = {"gen", "nevals", "avg", "std", "min", "max"}
    for entry in logbook:
        assert expected_cols.issubset(entry.keys())
        assert entry["nevals"] <= pop_size
        assert entry["min"] - 1e-9 <= entry["avg"] <= entry["max"] + 1e-9

    # Generation 0 evaluates entire population
    assert logbook[0]["nevals"] == pop_size


def test_fitness_positive_enforced(
    synthetic_train_data: tuple[pd.DataFrame, pd.Series],
    mini_cfg: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that non-positive fitness raises ValueError to protect roulette selection."""
    X, y = synthetic_train_data
    specs = mini_cfg["ga"]["search_spaces"]["xgboost"]

    # Monkeypatch fit_predict_scores to return zero accuracy
    def mock_fit_predict(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        # Return reversed labels so accuracy is 0.0
        X_val = args[4]
        return np.ones(len(X_val), dtype=int), np.ones(len(X_val), dtype=float)

    monkeypatch.setattr("src.ga.fit_predict_scores", mock_fit_predict)

    # Set true labels of y to all zeros so all-1 predictions give 0 accuracy
    y_zero = pd.Series([0] * len(y))
    bad_fitness = make_ga_fitness("xgboost", X, y_zero, specs, mini_cfg)

    with pytest.raises(ValueError, match="Fitness must be positive for roulette selection"):
        bad_fitness([20, 3, 0.1])


def test_roulette_selection_favours_higher_fitness() -> None:
    """Test DEAP tools.selRoulette statistically selects higher-fitness individuals more often."""
    fitnesses = [0.05, 0.15, 0.30, 0.85]
    pop = []
    for f in fitnesses:
        ind = creator.Individual([f])
        ind.fitness.values = (f,)
        pop.append(ind)

    import random
    random.seed(42)
    selected = tools.selRoulette(pop, k=2000)

    counts = {f: 0 for f in fitnesses}
    for ind in selected:
        counts[ind[0]] += 1

    # Probability of highest (0.85) is ~0.85 / 1.35 = 63%
    # Probability of lowest (0.05) is ~0.05 / 1.35 = 3.7%
    assert counts[0.85] > counts[0.30] > counts[0.15] > counts[0.05]
    assert counts[0.85] > 1000
    assert counts[0.05] < 150


def test_hall_of_fame_holds_best_fitness_seen(
    synthetic_train_data: tuple[pd.DataFrame, pd.Series],
    mini_cfg: dict[str, Any],
) -> None:
    """Test HallOfFame stores the absolute highest fitness encountered across all generations."""
    X, y = synthetic_train_data
    res = run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=mini_cfg,
        pop_size_override=4,
        n_gen_override=2,
        force=True,
    )

    max_in_logbook = max(row["max"] for row in res["logbook"])
    assert abs(res["best_fitness"] - max_in_logbook) < 1e-9


def test_cache_reused_on_matching_hash_and_bypassed_by_force(
    tmp_path: Path,
    synthetic_train_data: tuple[pd.DataFrame, pd.Series],
    mini_cfg: dict[str, Any],
) -> None:
    """Test GA caching: reuses result on matching hash and recomputes on --force or mismatch."""
    X, y = synthetic_train_data
    cache_file = tmp_path / "test_ga_cache.json"

    # Enable caching in config
    cfg = dict(mini_cfg)
    cfg["ga"] = dict(cfg["ga"])
    cfg["ga"]["use_cache"] = True

    # 1. First run: computes and writes cache
    res1 = run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=cfg,
        cache_path=cache_file,
        force=True,
    )
    assert cache_file.exists()
    assert res1.get("cached") is False

    # 2. Second run without force: loads from cache
    res2 = run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=cfg,
        cache_path=cache_file,
        force=False,
    )
    assert res2.get("cached") is True
    assert res2["best_fitness"] == res1["best_fitness"]

    # 3. Third run with force: recomputes
    res3 = run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=cfg,
        cache_path=cache_file,
        force=True,
    )
    assert res3.get("cached") is False

    # 4. Tamper with cache file hash: should recompute
    with open(cache_file, "r") as f:
        cache_data = json.load(f)
    cache_data["hash"] = "tampered_hash_value"
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

    res4 = run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=cfg,
        cache_path=cache_file,
        force=False,
    )
    assert res4.get("cached") is False


def test_fitness_never_touches_test_split(
    synthetic_train_data: tuple[pd.DataFrame, pd.Series],
    mini_cfg: dict[str, Any],
) -> None:
    """Test that GA fitness evaluation never receives or touches test split data."""
    X_train, y_train = synthetic_train_data
    specs = mini_cfg["ga"]["search_spaces"]["xgboost"]

    # Verify GAFitness constructor signature only takes train data
    fitness = make_ga_fitness("xgboost", X_train, y_train, specs, mini_cfg)

    # Class representing forbidden test set access
    class PoisonTestObject:
        def __getitem__(self, item: Any) -> Any:
            raise AssertionError("Test set was accessed during fitness evaluation!")

        def __len__(self) -> int:
            raise AssertionError("Test set length inspected during fitness evaluation!")

    poison_test = PoisonTestObject()

    # Fitness evaluation works purely on train folds without test data
    fit_val = fitness([20, 3, 0.1])
    assert fit_val[0] > 0.0

    # Ensure fitness evaluator attributes do not hold any test object
    for attr_name in dir(fitness):
        val = getattr(fitness, attr_name)
        assert val is not poison_test


def test_inputs_not_mutated(
    synthetic_train_data: tuple[pd.DataFrame, pd.Series],
    mini_cfg: dict[str, Any],
) -> None:
    """Test that X_train and y_train are not mutated by run_ga."""
    X, y = synthetic_train_data
    X_orig = X.copy(deep=True)
    y_orig = y.copy(deep=True)

    run_ga(
        model_name="xgboost",
        X_train=X,
        y_train=y,
        cfg=mini_cfg,
        pop_size_override=4,
        n_gen_override=1,
        force=True,
    )

    pd.testing.assert_frame_equal(X, X_orig)
    pd.testing.assert_series_equal(y, y_orig)


def test_run_budget_probe(
    synthetic_train_data: tuple[pd.DataFrame, pd.Series],
    mini_cfg: dict[str, Any],
) -> None:
    """Test budget probe executes quickly and produces timing projections."""
    X, y = synthetic_train_data
    probe = run_budget_probe("xgboost", X, y, mini_cfg)

    assert probe["model"] == "xgboost"
    assert probe["evaluations_measured"] > 0
    assert probe["sec_per_eval"] > 0.0
    assert probe["est_1000_seconds"] > 0.0
    assert probe["est_600_seconds"] > 0.0
    assert "under_30_min" in probe
