"""Unit tests for master ladder benchmark aggregation (Phase 2.0d).

All tests use synthetic mock results data in tmp_path; no heavy compute or raw dataset required.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.benchmark_ladder import (
    REQUIRED_RESULT_PATTERNS,
    find_latest_file,
    load_all_ladder_prerequisites,
    run_benchmark_ladder,
)


def _create_mock_metric(
    acc: float = 0.99,
    prec: float = 0.98,
    rec: float = 0.97,
    f1: float = 0.975,
    roc_auc: float = 0.999,
    pr_auc: float = 0.998,
    mcc: float = 0.97,
) -> dict:
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "n_test": 1803,
        "tn": 1500,
        "fp": 33,
        "fn": 8,
        "tp": 262,
    }


def _create_mock_suite(tmp_path: Path) -> dict[str, Path]:
    """Populate tmp_path with mock JSONs for all 7 required prerequisite files."""
    files: dict[str, Path] = {}

    # 1. Phase 1.8 master benchmark
    p18_payload = {
        "timestamp": "20261004_044733",
        "diagnostics": {
            "test_rows": 3065,
            "test_benign_rows": 1530,
            "test_suspicious_rows": 1535,
            "synthetic_test_count": 1245,
            "synthetic_test_ratio": 1245 / 3065,
            "majority_class_accuracy": 1530 / 3065,
        },
        "rows": [
            {
                "id": "xgboost_default",
                "paper_accuracy": 0.975,
                "paper_precision": 0.970,
                "paper_recall": 0.980,
                "paper_f1": 0.975,
                "paper_auc": 0.970,
                "pipeline_a_pso_accuracy": 0.9967,
                "pipeline_a_pso_precision": 0.9993,
                "pipeline_a_pso_recall": 0.9941,
                "pipeline_a_pso_f1": 0.9967,
                "pipeline_a_pso_auc": 0.9999,
            },
            {
                "id": "xgboost_ga",
                "paper_accuracy": 0.992,
                "paper_precision": 0.992,
                "paper_recall": 0.993,
                "paper_f1": 0.992,
                "paper_auc": 0.990,
                "pipeline_a_pso_accuracy": 0.9971,
                "pipeline_a_pso_precision": 1.0,
                "pipeline_a_pso_recall": 0.9941,
                "pipeline_a_pso_f1": 0.9971,
                "pipeline_a_pso_auc": 0.9999,
            },
            {
                "id": "svm_default",
                "paper_accuracy": 0.744,
                "paper_precision": 0.826,
                "paper_recall": 0.713,
                "paper_f1": 0.765,
                "paper_auc": 0.710,
                "pipeline_a_pso_accuracy": 0.9377,
                "pipeline_a_pso_precision": 0.9275,
                "pipeline_a_pso_recall": 0.9498,
                "pipeline_a_pso_f1": 0.9385,
                "pipeline_a_pso_auc": 0.9698,
            },
            {
                "id": "svm_ga",
                "paper_accuracy": 0.870,
                "paper_precision": 0.869,
                "paper_recall": 0.872,
                "paper_f1": 0.870,
                "paper_auc": 0.860,
                "pipeline_a_pso_accuracy": 0.9883,
                "pipeline_a_pso_precision": 0.9870,
                "pipeline_a_pso_recall": 0.9896,
                "pipeline_a_pso_f1": 0.9883,
                "pipeline_a_pso_auc": 0.9980,
            },
            {
                "id": "isolation_forest_default",
                "paper_accuracy": 0.694,
                "paper_precision": 0.732,
                "paper_recall": 0.683,
                "paper_f1": 0.707,
                "paper_auc": 0.530,
                "pipeline_a_pso_accuracy": 0.4395,
                "pipeline_a_pso_precision": 0.2269,
                "pipeline_a_pso_recall": 0.0495,
                "pipeline_a_pso_f1": 0.0813,
                "pipeline_a_pso_auc": 0.1869,
            },
            {
                "id": "isolation_forest_ga",
                "paper_accuracy": 0.824,
                "paper_precision": 0.824,
                "paper_recall": 0.827,
                "paper_f1": 0.825,
                "paper_auc": 0.790,
                "pipeline_a_pso_accuracy": 0.4897,
                "pipeline_a_pso_precision": 0.1463,
                "pipeline_a_pso_recall": 0.0039,
                "pipeline_a_pso_f1": 0.0076,
                "pipeline_a_pso_auc": 0.2289,
            },
            {
                "id": "cart_baseline",
                "paper_accuracy": 0.810,
                "paper_precision": 0.813,
                "paper_recall": 0.810,
                "paper_f1": 0.809,
                "paper_auc": 0.810,
                "pipeline_a_pso_accuracy": 0.9935,
                "pipeline_a_pso_precision": 0.9928,
                "pipeline_a_pso_recall": 0.9941,
                "pipeline_a_pso_f1": 0.9935,
                "pipeline_a_pso_auc": 0.9935,
            },
            {
                "id": "lof_baseline",
                "paper_accuracy": 0.949,
                "paper_precision": 0.946,
                "paper_recall": 0.954,
                "paper_f1": 0.949,
                "paper_auc": 0.940,
                "pipeline_a_pso_accuracy": 0.4349,
                "pipeline_a_pso_precision": 0.2909,
                "pipeline_a_pso_recall": 0.0893,
                "pipeline_a_pso_f1": 0.1366,
                "pipeline_a_pso_auc": 0.4012,
            },
        ],
    }
    f18 = tmp_path / "phase1_8_benchmark_20261004_044733.json"
    with open(f18, "w", encoding="utf-8") as f:
        json.dump(p18_payload, f)
    files["phase1_8_benchmark"] = f18

    # 2. Phase 2.0b ladder defaults
    split_rep = {
        "train_rows": 7209,
        "test_rows": 1803,
        "test_class_counts": {"0": 1533, "1": 270},
        "synthetic_test_count": 0,
        "synthetic_test_ratio": 0.0,
    }
    maj_rep = {
        "majority_class": 0,
        "accuracy": 1533 / 1803,
        "recall_suspicious": 0.0,
    }
    p20b_payload = {
        "timestamp": "20261004_161846",
        "rungs": {
            "l1": {
                "split_report": split_rep,
                "majority_baseline": maj_rep,
                "models": {
                    "xgboost": {"metrics": _create_mock_metric(acc=0.9945, f1=0.9816)},
                    "svm": {"metrics": _create_mock_metric(acc=0.9218, f1=0.7854)},
                    "isolation_forest": {"metrics": _create_mock_metric(acc=0.7349, f1=0.0478)},
                    "cart": {"metrics": _create_mock_metric(acc=0.9806, f1=0.9376)},
                    "lof": {"metrics": _create_mock_metric(acc=0.6916, f1=0.1775)},
                },
            },
            "b": {
                "split_report": split_rep,
                "majority_baseline": maj_rep,
                "models": {
                    "xgboost": {"metrics": _create_mock_metric(acc=0.9945, f1=0.9815)},
                    "svm": {"metrics": _create_mock_metric(acc=0.9196, f1=0.7800)},
                    "isolation_forest": {"metrics": _create_mock_metric(acc=0.7399, f1=0.0487)},
                    "cart": {"metrics": _create_mock_metric(acc=0.9828, f1=0.9441)},
                    "lof": {"metrics": _create_mock_metric(acc=0.7022, f1=0.1826)},
                },
            },
        },
    }
    f20b = tmp_path / "phase2_0b_pipeline_b_ladder_20261004_161846.json"
    with open(f20b, "w", encoding="utf-8") as f:
        json.dump(p20b_payload, f)
    files["phase2_0b_ladder"] = f20b

    # 3. Phase 2.0c GA L1 XGBoost
    p20c_l1_xgb = {
        "timestamp": "20261004_164511",
        "total_evaluations": 645,
        "elapsed_seconds": 150.2,
        "best_fitness_cv_f1_suspicious": 0.9951,
        "best_params": {"max_depth": 3, "learning_rate": 0.1},
        "test_metrics": _create_mock_metric(acc=0.9928, f1=0.9762),
    }
    f20c_l1_xgb = tmp_path / "phase2_0c_ga_l1_xgboost_20261004_164511.json"
    with open(f20c_l1_xgb, "w", encoding="utf-8") as f:
        json.dump(p20c_l1_xgb, f)
    files["phase2_0c_ga_l1_xgboost"] = f20c_l1_xgb

    # 4. Phase 2.0c GA B XGBoost
    p20c_b_xgb = {
        "timestamp": "20261004_164208",
        "total_evaluations": 643,
        "elapsed_seconds": 140.5,
        "best_fitness_cv_f1_suspicious": 0.9710,
        "best_params": {"max_depth": 4, "learning_rate": 0.2},
        "test_metrics": _create_mock_metric(acc=0.9911, f1=0.9706),
    }
    f20c_b_xgb = tmp_path / "phase2_0c_ga_b_xgboost_20261004_164208.json"
    with open(f20c_b_xgb, "w", encoding="utf-8") as f:
        json.dump(p20c_b_xgb, f)
    files["phase2_0c_ga_b_xgboost"] = f20c_b_xgb

    # 5. Phase 2.0c GA B SVM
    p20c_b_svm = {
        "timestamp": "20261004_171516",
        "total_evaluations": 625,
        "elapsed_seconds": 1253.88,
        "best_fitness_cv_f1_suspicious": 0.9243,
        "best_params": {"C": 80.5, "gamma": 4.2},
        "test_metrics": _create_mock_metric(acc=0.9795, f1=0.9350),
    }
    f20c_b_svm = tmp_path / "phase2_0c_ga_b_svm_20261004_171516.json"
    with open(f20c_b_svm, "w", encoding="utf-8") as f:
        json.dump(p20c_b_svm, f)
    files["phase2_0c_ga_b_svm"] = f20c_b_svm

    # 6. Phase 2.0c GA B Isolation Forest
    p20c_b_if = {
        "timestamp": "20261004_165411",
        "total_evaluations": 639,
        "elapsed_seconds": 320.1,
        "best_fitness_cv_f1_suspicious": 0.1314,
        "best_params": {"contamination": 0.4979, "n_estimators": 200},
        "test_metrics": _create_mock_metric(acc=0.3172, f1=0.0995),
    }
    f20c_b_if = tmp_path / "phase2_0c_ga_b_isolation_forest_20261004_165411.json"
    with open(f20c_b_if, "w", encoding="utf-8") as f:
        json.dump(p20c_b_if, f)
    files["phase2_0c_ga_b_isolation_forest"] = f20c_b_if

    # 7. Phase 2.0c duplicates
    p20c_dup = {
        "timestamp": "20261004_163639",
        "diagnostic_summary": {
            "duplicate_rows": {
                "redundant_rows_count": 266,
                "redundant_ratio": 266 / 9012,
                "redundant_by_class": {"0": 23, "1": 243},
                "cluster_rows_count": 483,
                "cluster_ratio": 483 / 9012,
                "cluster_by_class": {"0": 39, "1": 444},
            },
            "contradictory_groups": {"groups_count": 0, "rows_count": 0},
            "train_test_overlap": {
                "test_rows_in_train_count": 57,
                "test_rows_in_train_ratio": 57 / 1803,
                "test_rows_in_train_by_class": {"0": 9, "1": 48},
                "unseen_test_count": 1746,
                "unseen_test_ratio": 1746 / 1803,
                "unseen_test_by_class": {"0": 1524, "1": 222},
            },
        },
        "full_test_evaluation": {"metrics": _create_mock_metric(acc=0.9945, f1=0.9815)},
        "unseen_test_evaluation": {
            "metrics": _create_mock_metric(acc=0.9943, f1=0.9775),
            "majority_baseline": {"accuracy": 1524 / 1746, "recall_suspicious": 0.0},
        },
        "deltas_full_vs_unseen": {"accuracy": -0.0002, "f1": -0.0040},
    }
    f20c_dup = tmp_path / "phase2_0c_duplicates_20261004_163639.json"
    with open(f20c_dup, "w", encoding="utf-8") as f:
        json.dump(p20c_dup, f)
    files["phase2_0c_duplicates"] = f20c_dup

    return files


def test_missing_prerequisite_raises(tmp_path: Path):
    """Ensure FileNotFoundError is raised if any prerequisite file is missing."""
    # Empty directory
    with pytest.raises(FileNotFoundError, match="Missing required result file"):
        run_benchmark_ladder(tmp_path)


def test_missing_single_prerequisite_raises(tmp_path: Path):
    """Ensure missing even one file out of 7 raises FileNotFoundError."""
    files = _create_mock_suite(tmp_path)
    # Remove one file
    files["phase2_0c_duplicates"].unlink()

    with pytest.raises(FileNotFoundError, match="Missing required result file matching pattern 'phase2_0c_duplicates_"):
        run_benchmark_ladder(tmp_path)


def test_run_benchmark_ladder_generates_artifacts(tmp_path: Path):
    """Ensure run_benchmark_ladder produces valid json, csv, and md files."""
    _create_mock_suite(tmp_path)

    res = run_benchmark_ladder(tmp_path)
    assert res["out_json"].exists()
    assert res["out_csv"].exists()
    assert res["out_md"].exists()

    # Verify JSON content
    with open(res["out_json"], "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "timestamp" in payload
    assert "diagnostics" in payload
    assert "master_table" in payload
    assert "ga_summary" in payload
    assert "duplicate_diagnostics" in payload

    # Verify CSV has 10 data rows
    with open(res["out_csv"], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 10

    # Verify MD contains key sections and footnote
    md = res["markdown"]
    assert "# Master Leakage Ladder Benchmark (Phase 2.0d)" in md
    assert "Per-Rung Evaluation Distribution Diagnostics" in md
    assert "Master Leakage Ladder Benchmark Table" in md
    assert "Genetic Algorithm Optimization Summary" in md
    assert "Duplicate & Train/Test Overlap Diagnostic Summary" in md
    assert "Pipeline A's test set is balanced and contains synthetic rows" in md


def test_unrun_cells_marked_not_run(tmp_path: Path):
    """Ensure unrun combinations (e.g. CART GA, LOF GA, SVM GA on L1, IF GA on L1) are 'not run'."""
    _create_mock_suite(tmp_path)
    res = run_benchmark_ladder(tmp_path)
    master_table = {f"{r['model']}_{r['stage']}": r for r in res["payload"]["master_table"]}

    # CART GA should be not run across all rungs
    cart_ga = master_table["CART_GA Tuned"]
    assert cart_ga["rung_l1_accuracy"] == "not run"
    assert cart_ga["rung_b_accuracy"] == "not run"
    assert cart_ga["paper_accuracy"] == "not run"

    # LOF GA should be not run
    lof_ga = master_table["LOF_GA Tuned"]
    assert lof_ga["rung_l1_accuracy"] == "not run"
    assert lof_ga["rung_b_accuracy"] == "not run"

    # SVM GA on L1 was not run
    svm_ga = master_table["SVM_GA Tuned"]
    assert svm_ga["rung_l1_accuracy"] == "not run"
    assert svm_ga["rung_b_accuracy"] == pytest.approx(0.9795)

    # Isolation Forest GA on L1 was not run
    if_ga = master_table["Isolation Forest_GA Tuned"]
    assert if_ga["rung_l1_accuracy"] == "not run"
    assert if_ga["rung_b_accuracy"] == pytest.approx(0.3172)

    # Paper PR-AUC and MCC were not run
    xgb_def = master_table["XGBoost_Default"]
    assert xgb_def["paper_pr_auc"] == "not run"
    assert xgb_def["paper_mcc"] == "not run"


def test_values_match_input_files(tmp_path: Path):
    """Ensure numeric values in output table match input JSON fields."""
    _create_mock_suite(tmp_path)
    res = run_benchmark_ladder(tmp_path)
    master_table = {f"{r['model']}_{r['stage']}": r for r in res["payload"]["master_table"]}

    # XGBoost Default
    xgb_def = master_table["XGBoost_Default"]
    assert xgb_def["paper_accuracy"] == 0.975
    assert xgb_def["pipeline_a_accuracy"] == 0.9967
    assert xgb_def["rung_l1_accuracy"] == pytest.approx(0.9945)
    assert xgb_def["rung_b_accuracy"] == pytest.approx(0.9945)
    assert xgb_def["rung_b_f1"] == pytest.approx(0.9815)

    # XGBoost GA
    xgb_ga = master_table["XGBoost_GA Tuned"]
    assert xgb_ga["paper_accuracy"] == 0.992
    assert xgb_ga["pipeline_a_accuracy"] == 0.9971
    assert xgb_ga["rung_l1_accuracy"] == pytest.approx(0.9928)
    assert xgb_ga["rung_b_accuracy"] == pytest.approx(0.9911)
    assert xgb_ga["rung_b_f1"] == pytest.approx(0.9706)


def test_ga_summary_tuning_deltas(tmp_path: Path):
    """Ensure GA summary computes tuning delta as (test_f1 - best_inner_cv_fitness)."""
    _create_mock_suite(tmp_path)
    res = run_benchmark_ladder(tmp_path)
    ga_summary = {f"{r['model']}_{r['rung']}": r for r in res["payload"]["ga_summary"]}

    xgb_b = ga_summary["XGBoost_Rung B"]
    assert xgb_b["best_inner_cv_fitness"] == pytest.approx(0.9710)
    assert xgb_b["test_f1"] == pytest.approx(0.9706)
    assert xgb_b["tuning_vs_test_delta"] == pytest.approx(0.9706 - 0.9710)

    xgb_l1 = ga_summary["XGBoost_Rung L1"]
    assert xgb_l1["best_inner_cv_fitness"] == pytest.approx(0.9951)
    assert xgb_l1["test_f1"] == pytest.approx(0.9762)
    assert xgb_l1["tuning_vs_test_delta"] == pytest.approx(0.9762 - 0.9951)


def test_per_rung_diagnostics(tmp_path: Path):
    """Ensure per-rung diagnostics correctly capture test size and synthetic ratio."""
    _create_mock_suite(tmp_path)
    res = run_benchmark_ladder(tmp_path)
    diags = res["payload"]["diagnostics"]

    assert diags["pipeline_a"]["test_size"] == 3065
    assert diags["pipeline_a"]["synthetic_fraction"] == pytest.approx(1245 / 3065)

    assert diags["rung_l1"]["test_size"] == 1803
    assert diags["rung_l1"]["synthetic_fraction"] == 0.0

    assert diags["rung_b"]["test_size"] == 1803
    assert diags["rung_b"]["synthetic_fraction"] == 0.0
