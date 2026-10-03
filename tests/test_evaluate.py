"""Tests for evaluation metrics, consistency checks, and paper confusion matrix validation (src/evaluate.py).

All tests use synthetic data and paper fixtures; no raw dataset required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    check_consistency,
    compute_metrics,
    majority_baseline,
    paper_layout_check,
    write_results,
)


def test_compute_metrics_known_matrix() -> None:
    """Verify compute_metrics on a known small synthetic ground truth and prediction pair."""
    # Ground truth: 4 benign (0), 6 suspicious (1)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    # Predictions:
    # 0s: 3 correct (TN), 1 incorrect (FP) -> tn=3, fp=1
    # 1s: 4 correct (TP), 2 incorrect (FN) -> tp=4, fn=2
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.85, 0.65, 0.4, 0.35])

    res = compute_metrics(y_true, y_pred, scores=scores)

    assert res["tn"] == 3
    assert res["fp"] == 1
    assert res["fn"] == 2
    assert res["tp"] == 4
    assert res["n_test"] == 10
    assert res["class_counts"] == {0: 4, 1: 6}
    assert res["confusion_matrix"] == [[3, 1], [2, 4]]

    # Accuracy: (4 + 3) / 10 = 0.7
    assert np.isclose(res["accuracy"], 0.7)
    # MAE: (1 + 2) / 10 = 0.3 == 1 - accuracy
    assert np.isclose(res["mae"], 0.3)
    assert np.isclose(res["mae"], 1.0 - res["accuracy"])

    # Suspicious class (1): precision = 4 / (4 + 1) = 0.8, recall = 4 / (4 + 2) = 0.6667
    assert np.isclose(res["precision"], 4.0 / 5.0)
    assert np.isclose(res["recall"], 4.0 / 6.0)
    expected_f1_susp = 2 * (0.8 * (4.0 / 6.0)) / (0.8 + 4.0 / 6.0)
    assert np.isclose(res["f1"], expected_f1_susp)

    # Benign class (0): precision = 3 / (3 + 2) = 0.6, recall = 3 / (3 + 1) = 0.75
    assert np.isclose(res["precision_benign"], 3.0 / 5.0)
    assert np.isclose(res["recall_benign"], 3.0 / 4.0)

    # Macro averages
    assert np.isclose(res["macro_precision"], (0.8 + 0.6) / 2.0)
    assert np.isclose(res["macro_recall"], (4.0 / 6.0 + 0.75) / 2.0)

    # AUC and MCC
    assert res["roc_auc"] is not None and 0.0 <= res["roc_auc"] <= 1.0
    assert res["pr_auc"] is not None and 0.0 <= res["pr_auc"] <= 1.0
    assert -1.0 <= res["mcc"] <= 1.0


def test_compute_metrics_degenerate_cases_do_not_raise() -> None:
    """Verify compute_metrics handles degenerate cases (all 0s or all 1s predicted) gracefully."""
    y_true = np.array([0, 0, 1, 1])

    # All 0s predicted
    pred_all_0 = np.array([0, 0, 0, 0])
    res_0 = compute_metrics(y_true, pred_all_0)
    assert res_0["accuracy"] == 0.5
    assert res_0["precision"] == 0.0  # tp=0, fp=0 -> 0.0 without division error
    assert res_0["recall"] == 0.0
    assert res_0["f1"] == 0.0
    assert res_0["roc_auc"] is None

    # All 1s predicted
    pred_all_1 = np.array([1, 1, 1, 1])
    res_1 = compute_metrics(y_true, pred_all_1)
    assert res_1["accuracy"] == 0.5
    assert res_1["precision_benign"] == 0.0  # tn=0, fn=0 -> 0.0 without division error
    assert res_1["recall_benign"] == 0.0


@pytest.mark.parametrize(
    "name,tp,fp,tn,fn,reported",
    [
        (
            "xgboost_default",
            1498,
            46,
            1491,
            30,
            {"accuracy": 0.975, "mae": 0.03, "precision": 0.97, "recall": 0.98, "f1": 0.975, "auc": 0.97},
        ),
        (
            "svm_default",
            1275,
            269,
            1008,
            513,
            {"accuracy": 0.744, "mae": 0.26, "precision": 0.826, "recall": 0.713, "f1": 0.765, "auc": 0.71},
        ),
        (
            "if_default",
            1130,
            414,
            997,
            524,
            {"accuracy": 0.694, "mae": 0.30, "precision": 0.732, "recall": 0.683, "f1": 0.707, "auc": 0.53},
        ),
        (
            "xgboost_ga",
            1531,
            13,
            1510,
            11,
            {"accuracy": 0.992, "mae": 0.01, "precision": 0.992, "recall": 0.993, "f1": 0.992, "auc": 0.99},
        ),
        (
            "svm_ga",
            1341,
            203,
            1325,
            196,
            {"accuracy": 0.870, "mae": 0.13, "precision": 0.869, "recall": 0.872, "f1": 0.870, "auc": 0.86},
        ),
        (
            "if_ga",
            1273,
            271,
            1254,
            267,
            {"accuracy": 0.824, "mae": 0.18, "precision": 0.824, "recall": 0.827, "f1": 0.825, "auc": 0.79},
        ),
    ],
)
def test_paper_layout_check_on_all_six_matrices(
    name: str, tp: int, fp: int, tn: int, fn: int, reported: dict[str, float]
) -> None:
    """Verify paper_layout_check matches reported metrics within 0.002 and checks printed row sums."""
    res = paper_layout_check(tp, fp, tn, fn, reported=reported)

    # 1. Total samples must be 3065
    assert res["total_n"] == 3065

    # 2. Invariant printed row sums across all six matrices
    assert res["reading_printed_rows"]["tp_plus_fp"] == 1544
    assert res["reading_printed_rows"]["tn_plus_fn"] == 1521

    # 3. Accuracy, precision, recall, F1 within 0.002 of reported values
    comp = res["computed_metrics"]
    assert abs(comp["accuracy"] - reported["accuracy"]) <= 0.002
    assert abs(comp["precision"] - reported["precision"]) <= 0.002
    assert abs(comp["recall"] - reported["recall"]) <= 0.002
    assert abs(comp["f1"] - reported["f1"]) <= 0.002

    # 4. MAE in paper is rounded to 2 decimal places (or 1 where trailing zero dropped),
    # so tolerance is relaxed to 0.007
    assert abs(comp["mae"] - reported["mae"]) <= 0.007


def test_check_consistency_passes_and_flags_mismatches() -> None:
    """Verify check_consistency passes on consistent models and flags discrepancies."""
    consistent_results = {
        "model_a": {"tp": 10, "fn": 2, "fp": 1, "tn": 7, "accuracy": 17.0 / 20.0, "n_test": 20, "class_counts": {0: 8, 1: 12}},
        "model_b": {"tp": 11, "fn": 1, "fp": 2, "tn": 6, "accuracy": 17.0 / 20.0, "n_test": 20, "class_counts": {0: 8, 1: 12}},
    }

    checks_pass = check_consistency(
        consistent_results,
        expected_counts={"n_test": 20, 0: 8, 1: 12},
    )
    for c in checks_pass:
        assert c["passed"] is True

    # Deliberately inconsistent model
    inconsistent_results = {
        "model_a": {"tp": 10, "fn": 2, "fp": 1, "tn": 7, "accuracy": 17.0 / 20.0, "n_test": 20, "class_counts": {0: 8, 1: 12}},
        "model_c": {"tp": 8, "fn": 10, "fp": 1, "tn": 1, "accuracy": 0.99, "n_test": 20, "class_counts": {0: 2, 1: 18}},
    }

    checks_fail = check_consistency(
        inconsistent_results,
        expected_counts={"n_test": 20, 0: 8, 1: 12},
    )
    passed_map = {c["name"]: c["passed"] for c in checks_fail}
    assert passed_map["class_totals_consistent_across_models"] is False
    assert passed_map["accuracy_recomputed_matches_reported"] is False
    assert passed_map["test_counts_match_expected"] is False


def test_majority_baseline_known_vector() -> None:
    """Verify majority_baseline on a known vector."""
    # 7 benign (0), 3 suspicious (1)
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    res = majority_baseline(y_true)

    assert res["majority_class"] == 0
    assert np.isclose(res["accuracy"], 0.7)
    assert res["recall_suspicious"] == 0.0


def test_write_results(tmp_path: pytest.MonkeyPatch) -> None:
    """Verify write_results creates expected JSON structure."""
    cfg = {"paths": {"results": str(tmp_path)}, "seed": 42}
    payload = {"data": {"val": 123}}
    timings = {"step1": 0.05}

    out_file = write_results("testphase", "testrun", payload, cfg, timings)
    assert out_file.exists()
    assert out_file.suffix == ".json"
