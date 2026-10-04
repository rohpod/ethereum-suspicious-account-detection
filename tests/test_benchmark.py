"""Unit tests for master benchmark table aggregation and validation (Phase 1.8).

All tests use synthetic mock results data in tmp_path; no raw dataset required.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.benchmark import (
    REQUIRED_RESULT_PATTERNS,
    find_latest_file,
    generate_benchmark,
    load_all_prerequisite_results,
    perform_cross_model_checks,
)


def _create_mock_metric(
    n_test: int = 3065,
    tn: int = 1530,
    fp: int = 0,
    fn: int = 10,
    tp: int = 1525,
    acc: float = 0.9967,
) -> dict:
    """Helper to create a consistent mock metric dictionary."""
    return {
        "n_test": n_test,
        "class_counts": {0: tn + fp, 1: fn + tp},
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": acc,
        "mae": 1.0 - acc,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "f1": 2.0 * tp / (2.0 * tp + fp + fn),
        "roc_auc": 0.999,
        "pr_auc": 0.999,
    }


def _create_mock_suite(tmp_path: Path) -> dict[str, Path]:
    """Helper to populate tmp_path with valid mock result files for all 5 prerequisites."""
    cfg_snapshot = {"seed": 42}
    files = {}

    # 1. Phase 1.5 defaults
    m15 = _create_mock_metric()
    p15_payload = {
        "timestamp": "20261004_031101",
        "config": cfg_snapshot,
        "results": {
            "feature_sets": {
                "pso": {
                    "feature_count": 22,
                    "split_report": {
                        "train_rows": 12259,
                        "test_rows": 3065,
                        "test_class_counts": {0: 1530, 1: 1535},
                        "synthetic_test_count": 1245,
                        "synthetic_test_ratio": 1245 / 3065,
                    },
                    "models": {
                        "xgboost": {"metrics": m15},
                        "svm": {"metrics": m15},
                        "isolation_forest": {"metrics": m15},
                    },
                },
                "table2": {
                    "feature_count": 14,
                    "split_report": {
                        "train_rows": 12259,
                        "test_rows": 3065,
                        "test_class_counts": {0: 1530, 1: 1535},
                        "synthetic_test_count": 1245,
                        "synthetic_test_ratio": 1245 / 3065,
                    },
                    "models": {
                        "xgboost": {"metrics": m15},
                        "svm": {"metrics": m15},
                        "isolation_forest": {"metrics": m15},
                    },
                },
            }
        },
    }
    f15 = tmp_path / "phase1_5_pipeline_a_defaults_20261004_031101.json"
    with open(f15, "w") as f:
        json.dump(p15_payload, f)
    files["phase1_5_defaults"] = f15

    # 2. Phase 1.6 GA XGBoost
    f16_xgb = tmp_path / "phase1_6_ga_xgboost_20261004_034834.json"
    with open(f16_xgb, "w") as f:
        json.dump({"timestamp": "20261004_034834", "test_metrics": _create_mock_metric()}, f)
    files["phase1_6_ga_xgboost"] = f16_xgb

    # 3. Phase 1.6 GA SVM
    f16_svm = tmp_path / "phase1_6_ga_svm_20261004_042747.json"
    with open(f16_svm, "w") as f:
        json.dump({"timestamp": "20261004_042747", "test_metrics": _create_mock_metric()}, f)
    files["phase1_6_ga_svm"] = f16_svm

    # 4. Phase 1.6 GA IF
    f16_if = tmp_path / "phase1_6_ga_isolation_forest_20261004_035452.json"
    with open(f16_if, "w") as f:
        json.dump({"timestamp": "20261004_035452", "test_metrics": _create_mock_metric()}, f)
    files["phase1_6_ga_isolation_forest"] = f16_if

    # 5. Phase 1.7 Baselines
    p17_payload = {
        "timestamp": "20261004_050000",
        "results": {
            "feature_sets": {
                "pso": {
                    "models": {
                        "cart": {"metrics": _create_mock_metric()},
                        "lof": {"metrics": _create_mock_metric()},
                    }
                },
                "table2": {
                    "models": {
                        "cart": {"metrics": _create_mock_metric()},
                        "lof": {"metrics": _create_mock_metric()},
                    }
                },
            }
        },
    }
    f17 = tmp_path / "phase1_7_baselines_20261004_050000.json"
    with open(f17, "w") as f:
        json.dump(p17_payload, f)
    files["phase1_7_baselines"] = f17

    return files


def test_find_latest_file(tmp_path: Path) -> None:
    """Verify find_latest_file picks the lexicographically latest timestamp."""
    p1 = tmp_path / "test_run_20260101_000000.json"
    p2 = tmp_path / "test_run_20260102_000000.json"
    p1.touch()
    p2.touch()

    latest = find_latest_file(tmp_path, "test_run_*.json")
    assert latest == p2


def test_find_latest_file_missing_raises_error(tmp_path: Path) -> None:
    """Verify find_latest_file raises FileNotFoundError with descriptive message."""
    with pytest.raises(FileNotFoundError, match="Missing required result file"):
        find_latest_file(tmp_path, "nonexistent_pattern_*.json")


def test_load_all_prerequisite_results_detects_missing(tmp_path: Path) -> None:
    """Verify load_all_prerequisite_results raises if any of the 5 files is absent."""
    # Create 4 of 5 files
    _create_mock_suite(tmp_path)
    (tmp_path / "phase1_7_baselines_20261004_050000.json").unlink()

    import re

    with pytest.raises(FileNotFoundError, match=re.escape(REQUIRED_RESULT_PATTERNS["phase1_7_baselines"])):
        load_all_prerequisite_results(tmp_path)


def test_cross_model_checks_pass_and_detect_inconsistency() -> None:
    """Verify perform_cross_model_checks validates consistent totals and detects injected errors."""
    m_valid = _create_mock_metric(n_test=3065, tn=1530, fp=0, fn=10, tp=1525, acc=3055/3065)
    metrics_dict = {
        "m1": m_valid,
        "m2": m_valid,
    }
    checks_pass = perform_cross_model_checks(metrics_dict, expected_n_test=3065)
    assert all(c["passed"] for c in checks_pass)

    # Injected inconsistency: different class balance (pos total 1510 != 1535)
    m_inconsistent = _create_mock_metric(n_test=3065, tn=1530, fp=25, fn=10, tp=1500, acc=3030/3065)
    metrics_inconsistent = {
        "m1": m_valid,
        "m2": m_inconsistent,
    }
    checks_fail = perform_cross_model_checks(metrics_inconsistent, expected_n_test=3065)
    has_failed = any(not c["passed"] for c in checks_fail)
    assert has_failed


def test_generate_benchmark_creates_outputs(tmp_path: Path) -> None:
    """Verify generate_benchmark creates valid JSON, CSV, and MD artifacts."""
    _create_mock_suite(tmp_path)

    cfg = {
        "paths": {"results": str(tmp_path)},
        "paper_results": {
            "default": {"xgboost": {"accuracy": 0.975, "f1": 0.975, "auc": 0.97}},
            "ga": {"xgboost": {"accuracy": 0.992, "f1": 0.992, "auc": 0.99}},
            "table10": {"cart": {"accuracy": 0.810}, "lof": {"accuracy": 0.949}},
        },
    }

    result = generate_benchmark(cfg=cfg, results_dir=tmp_path)

    # Output paths exist
    assert result["out_json"].exists()
    assert result["out_csv"].exists()
    assert result["out_md"].exists()

    # JSON content validation
    with open(result["out_json"], "r") as f:
        json_data = json.load(f)
    assert "timestamp" in json_data
    assert "diagnostics" in json_data
    assert "consistency_checks" in json_data
    assert len(json_data["rows"]) == 9

    # CSV validation
    with open(result["out_csv"], "r") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 9
    for r in reader:
        assert r["pipeline_b"] == "not yet run"
        assert r["optimised"] == "not yet run"

    # Markdown validation
    with open(result["out_md"], "r") as f:
        md_text = f.read()
    assert "# Master Benchmark: Paper vs Pipeline A (Phase 1.8)" in md_text
    assert "40.62%" in md_text
    assert "| Model |" in md_text
    assert "Table 10 Baseline Observations" in md_text
