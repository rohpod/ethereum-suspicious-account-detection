"""Evaluation harness, performance metrics, paper confusion matrix validation, and result writing.

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
  Equations 12-15 (Accuracy, Recall, Precision, F1 Score)
  Section 5 & Tables 8-9 (Baseline and GA confusion matrices & evaluation metrics)
  Figures 9-14 (Confusion matrices before and after GA)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification and anomaly detection metrics.

    Calculates standard metrics with suspicious accounts treated as class 1:
    - Confusion matrix in standard scikit-learn orientation [[TN, FP], [FN, TP]]
    - Accuracy, MAE (mean absolute error of hard labels: 1 - accuracy)
    - Precision, recall, and F1 for the suspicious class (class 1)
    - Precision, recall, and F1 for the benign class (class 0)
    - Macro-averaged precision, recall, and F1
    - Matthews correlation coefficient (MCC)
    - ROC-AUC and PR-AUC (Average Precision) if scores are provided

    Handles degenerate cases (e.g. all one class predicted) without raising.

    Args:
        y_true: True binary labels (0 = benign, 1 = suspicious).
        y_pred: Predicted binary labels (0 = benign, 1 = suspicious).
        scores: Optional continuous risk/anomaly scores (higher = more suspicious).

    Returns:
        dict containing all computed performance metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    n_test = len(y_true_arr)
    class_0_count = int((y_true_arr == 0).sum())
    class_1_count = int((y_true_arr == 1).sum())

    # Confusion matrix: [[TN, FP], [FN, TP]]
    # labels=[0, 1] ensures 2x2 matrix even in degenerate cases
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    total = tn + fp + fn + tp
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    mae = float((fp + fn) / total) if total > 0 else 0.0

    # Suspicious class (1) metrics
    p_denom_susp = tp + fp
    precision_susp = float(tp / p_denom_susp) if p_denom_susp > 0 else 0.0

    r_denom_susp = tp + fn
    recall_susp = float(tp / r_denom_susp) if r_denom_susp > 0 else 0.0

    f1_denom_susp = precision_susp + recall_susp
    f1_susp = (
        float(2.0 * precision_susp * recall_susp / f1_denom_susp)
        if f1_denom_susp > 0
        else 0.0
    )

    # Benign class (0) metrics
    p_denom_benign = tn + fn
    precision_benign = float(tn / p_denom_benign) if p_denom_benign > 0 else 0.0

    r_denom_benign = tn + fp
    recall_benign = float(tn / r_denom_benign) if r_denom_benign > 0 else 0.0

    f1_denom_benign = precision_benign + recall_benign
    f1_benign = (
        float(2.0 * precision_benign * recall_benign / f1_denom_benign)
        if f1_denom_benign > 0
        else 0.0
    )

    # Macro averages
    macro_precision = float((precision_susp + precision_benign) / 2.0)
    macro_recall = float((recall_susp + recall_benign) / 2.0)
    macro_f1 = float((f1_susp + f1_benign) / 2.0)

    # Matthews correlation coefficient
    mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))

    # ROC-AUC and PR-AUC
    roc_auc = None
    pr_auc = None
    if scores is not None:
        scores_arr = np.asarray(scores, dtype=float)
        # Both classes must be present in y_true to compute AUC
        if len(np.unique(y_true_arr)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true_arr, scores_arr))
            except ValueError:
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(y_true_arr, scores_arr))
            except ValueError:
                pr_auc = None

    return {
        "n_test": n_test,
        "class_counts": {0: class_0_count, 1: class_1_count},
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "mae": mae,
        "precision": precision_susp,
        "recall": recall_susp,
        "f1": f1_susp,
        "precision_benign": precision_benign,
        "recall_benign": recall_benign,
        "f1_benign": f1_benign,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def paper_layout_check(
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    reported: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Recompute metrics using the paper's formulas on the cells as printed in Tables 8-9.

    Paper Reference: Equations 12-15 & Tables 8-9.
    Eq. 12: Accuracy = (TP + TN) / (TP + FP + TN + FN)
    Eq. 13: Recall = TP / (TP + FN)
    Eq. 14: Precision = TP / (TP + FP)
    Eq. 15: F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

    Exposes sums under both interpretations:
    - Reading 1 (Standard row totals): TP+FN (actual pos), FP+TN (actual neg)
    - Reading 2 (Printed row totals): TP+FP (1,544), TN+FN (1,521)

    Args:
        tp: Cell value labeled 'TP' in paper.
        fp: Cell value labeled 'FP' in paper.
        tn: Cell value labeled 'TN' in paper.
        fn: Cell value labeled 'FN' in paper.
        reported: Optional dict of metrics reported in paper (accuracy, mae, precision, recall, f1, auc).

    Returns:
        dict containing sums, recomputed metrics, and deltas to reported values.
    """
    total_n = tp + fp + tn + fn

    sum_tp_fp = tp + fp
    sum_tn_fn = tn + fn
    sum_tp_fn = tp + fn
    sum_fp_tn = fp + tn

    accuracy = (tp + tn) / total_n if total_n > 0 else 0.0
    mae = (fp + fn) / total_n if total_n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    computed = {
        "accuracy": accuracy,
        "mae": mae,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    deltas = {}
    if reported:
        for k in ["accuracy", "mae", "precision", "recall", "f1"]:
            if k in reported:
                deltas[k] = float(computed[k] - reported[k])

    return {
        "cells": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total_n": total_n,
        "reading_printed_rows": {
            "tp_plus_fp": sum_tp_fp,
            "tn_plus_fn": sum_tn_fn,
        },
        "reading_actual_classes": {
            "tp_plus_fn": sum_tp_fn,
            "fp_plus_tn": sum_fp_tn,
        },
        "computed_metrics": computed,
        "reported_metrics": reported,
        "deltas": deltas,
    }


def majority_baseline(y_true: np.ndarray | pd.Series) -> dict[str, Any]:
    """Calculate majority-class baseline performance for an always-benign classifier.

    Args:
        y_true: True binary labels (0 = benign, 1 = suspicious).

    Returns:
        dict containing majority class, accuracy, and suspicious class recall (0.0).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    total = len(y_true_arr)
    benign_count = int((y_true_arr == 0).sum())

    acc = float(benign_count / total) if total > 0 else 0.0

    return {
        "majority_class": 0,
        "accuracy": acc,
        "recall_suspicious": 0.0,
    }


def check_consistency(
    results: dict[str, dict[str, Any]],
    expected_counts: dict[str, int] | None = None,
    accuracy_tolerance: float = 1e-3,
) -> list[dict[str, Any]]:
    """Verify consistency across multiple model evaluations on the same test set.

    Checks:
    (a) Constant test set class balance across models: TP+FN and FP+TN identical
    (b) Accuracy recomputed from confusion matrix matches reported accuracy
    (c) n_test and class counts match expected_counts if specified

    Never raises; returns check records with passed status and details.

    Args:
        results: Dictionary mapping model names to their metric dictionaries.
        expected_counts: Optional dict with 'n_test', 0, 1 keys.
        accuracy_tolerance: Max allowable difference between recomputed and reported accuracy.

    Returns:
        list of check result dicts: [{'name': ..., 'passed': bool, 'detail': ...}]
    """
    checks: list[dict[str, Any]] = []

    # Check (a): TP+FN and FP+TN identical across all models
    class_totals: dict[str, tuple[int, int]] = {}
    for model_name, res in results.items():
        if all(k in res for k in ["tp", "fn", "fp", "tn"]):
            pos = res["tp"] + res["fn"]
            neg = res["fp"] + res["tn"]
            class_totals[model_name] = (pos, neg)

    if len(class_totals) > 1:
        _, first_totals = next(iter(class_totals.items()))
        mismatched = {
            m: t for m, t in class_totals.items() if t != first_totals
        }
        if not mismatched:
            checks.append({
                "name": "class_totals_consistent_across_models",
                "passed": True,
                "detail": f"All {len(class_totals)} models evaluated on identical class totals: positive={first_totals[0]}, negative={first_totals[1]}",
            })
        else:
            checks.append({
                "name": "class_totals_consistent_across_models",
                "passed": False,
                "detail": f"Inconsistent test set totals found across models: {class_totals}",
            })

    # Check (b): Recomputed accuracy equals reported accuracy
    acc_mismatches: list[str] = []
    for model_name, res in results.items():
        if all(k in res for k in ["tp", "tn", "fp", "fn", "accuracy"]):
            tot = res["tp"] + res["tn"] + res["fp"] + res["fn"]
            recomputed_acc = (res["tp"] + res["tn"]) / tot if tot > 0 else 0.0
            reported_acc = res["accuracy"]
            diff = abs(recomputed_acc - reported_acc)
            if diff > accuracy_tolerance:
                acc_mismatches.append(
                    f"{model_name}: reported={reported_acc:.4f}, recomputed={recomputed_acc:.4f}, diff={diff:.4f}"
                )

    if not acc_mismatches:
        checks.append({
            "name": "accuracy_recomputed_matches_reported",
            "passed": True,
            "detail": f"All models have recomputed accuracy matching reported accuracy within tolerance {accuracy_tolerance}.",
        })
    else:
        checks.append({
            "name": "accuracy_recomputed_matches_reported",
            "passed": False,
            "detail": f"Accuracy discrepancies exceeding tolerance {accuracy_tolerance}: {'; '.join(acc_mismatches)}",
        })

    # Check (c): Test counts match expected_counts if provided
    if expected_counts:
        exp_n = expected_counts.get("n_test")
        exp_0 = expected_counts.get(0)
        exp_1 = expected_counts.get(1)

        counts_mismatches = []
        for model_name, res in results.items():
            if exp_n is not None and res.get("n_test") != exp_n:
                counts_mismatches.append(
                    f"{model_name}: n_test={res.get('n_test')} != expected {exp_n}"
                )
            actual_counts = res.get("class_counts", {})
            if exp_0 is not None and actual_counts.get(0) != exp_0:
                counts_mismatches.append(
                    f"{model_name}: class 0 count={actual_counts.get(0)} != expected {exp_0}"
                )
            if exp_1 is not None and actual_counts.get(1) != exp_1:
                counts_mismatches.append(
                    f"{model_name}: class 1 count={actual_counts.get(1)} != expected {exp_1}"
                )

        if not counts_mismatches:
            checks.append({
                "name": "test_counts_match_expected",
                "passed": True,
                "detail": f"All models match expected test set counts: {expected_counts}",
            })
        else:
            checks.append({
                "name": "test_counts_match_expected",
                "passed": False,
                "detail": f"Counts mismatch: {'; '.join(counts_mismatches)}",
            })

    return checks


def write_results(
    phase: str,
    name: str,
    payload: dict[str, Any],
    cfg: dict[str, Any],
    timings: dict[str, float],
) -> Path:
    """Write standardized experiment results to results/<phase>_<name>_<timestamp>.json.

    Mirrors the JSON structure established in Phase 1.1 and 1.2:
    - timestamp: UTC formatted string (YYYYMMDD_HHMMSS)
    - config: Configuration dictionary snapshot
    - Payload fields merged at top level
    - timings_seconds: Per-stage wall-clock timings

    Args:
        phase: Phase prefix string (e.g. 'phase1_3').
        name: Experiment descriptor (e.g. 'paper_confusion_check').
        payload: Experiment-specific data dictionary.
        cfg: Configuration dictionary snapshot.
        timings: Dictionary of stage names to execution times in seconds.

    Returns:
        Path to written results JSON file.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = Path(cfg.get("paths", {}).get("results", "results/"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{phase}_{name}_{timestamp}.json"

    result_payload: dict[str, Any] = {
        "timestamp": timestamp,
        "config": cfg,
        **payload,
        "timings_seconds": timings,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    return out_file


if __name__ == "__main__":
    t0 = time.time()
    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Paper confusion matrix fixtures from Tables 8 & 9 and Figures 9-14
    paper_matrices = {
        "xgboost_default": {
            "tp": 1498,
            "fp": 46,
            "tn": 1491,
            "fn": 30,
            "reported": {
                "accuracy": 0.975,
                "mae": 0.03,
                "precision": 0.97,
                "recall": 0.98,
                "f1": 0.975,
                "auc": 0.97,
            },
        },
        "svm_default": {
            "tp": 1275,
            "fp": 269,
            "tn": 1008,
            "fn": 513,
            "reported": {
                "accuracy": 0.744,
                "mae": 0.26,
                "precision": 0.826,
                "recall": 0.713,
                "f1": 0.765,
                "auc": 0.71,
            },
        },
        "if_default": {
            "tp": 1130,
            "fp": 414,
            "tn": 997,
            "fn": 524,
            "reported": {
                "accuracy": 0.694,
                "mae": 0.30,
                "precision": 0.732,
                "recall": 0.683,
                "f1": 0.707,
                "auc": 0.53,
            },
        },
        "xgboost_ga": {
            "tp": 1531,
            "fp": 13,
            "tn": 1510,
            "fn": 11,
            "reported": {
                "accuracy": 0.992,
                "mae": 0.01,
                "precision": 0.992,
                "recall": 0.993,
                "f1": 0.992,
                "auc": 0.99,
            },
        },
        "svm_ga": {
            "tp": 1341,
            "fp": 203,
            "tn": 1325,
            "fn": 196,
            "reported": {
                "accuracy": 0.870,
                "mae": 0.13,
                "precision": 0.869,
                "recall": 0.872,
                "f1": 0.870,
                "auc": 0.86,
            },
        },
        "if_ga": {
            "tp": 1273,
            "fp": 271,
            "tn": 1254,
            "fn": 267,
            "reported": {
                "accuracy": 0.824,
                "mae": 0.18,
                "precision": 0.824,
                "recall": 0.827,
                "f1": 0.825,
                "auc": 0.79,
            },
        },
    }

    t_check_start = time.time()
    matrix_checks: dict[str, Any] = {}
    tp_plus_fp_list: list[int] = []
    tn_plus_fn_list: list[int] = []

    for name, fixture in paper_matrices.items():
        check_res = paper_layout_check(
            tp=fixture["tp"],
            fp=fixture["fp"],
            tn=fixture["tn"],
            fn=fixture["fn"],
            reported=fixture["reported"],
        )
        matrix_checks[name] = check_res
        tp_plus_fp_list.append(check_res["reading_printed_rows"]["tp_plus_fp"])
        tn_plus_fn_list.append(check_res["reading_printed_rows"]["tn_plus_fn"])

    t_check = time.time() - t_check_start

    # Summary of findings
    all_tp_fp_constant = len(set(tp_plus_fp_list)) == 1 and tp_plus_fp_list[0] == 1544
    all_tn_fn_constant = len(set(tn_plus_fn_list)) == 1 and tn_plus_fn_list[0] == 1521

    summary = {
        "all_tp_plus_fp_equal_1544": all_tp_fp_constant,
        "all_tn_plus_fn_equal_1521": all_tn_fn_constant,
        "total_test_samples": 3065,
        "explanation": (
            "Under reading 1 (standard actual-class totals TP+FN and FP+TN), class counts "
            "fluctuate across models (e.g. suspicious totals range from 1528 to 1788). "
            "Under reading 2 (printed row totals TP+FP and TN+FN), sums are strictly invariant "
            "across all six models: TP+FP == 1544 and TN+FN == 1521 (total 3065). "
            "This proves the test set was identical (1544 benign, 1521 suspicious per Section 5), "
            "but the authors interpreted scikit-learn's confusion matrix [[TN, FP], [FN, TP]] "
            "as [[TP, FP], [FN, TN]], explaining the apparent test set inconsistency in Section 3 item 5."
        ),
    }

    t_total = time.time() - t0

    payload = {
        "paper_confusion_matrix_checks": matrix_checks,
        "summary": summary,
    }

    timings = {
        "paper_layout_check": t_check,
        "total": t_total,
    }

    out_file = write_results(
        phase="phase1_3",
        name="paper_confusion_check",
        payload=payload,
        cfg=cfg,
        timings=timings,
    )

    print(f"Paper confusion check completed. Results written to: {out_file}")
    print(f"TP+FP == 1544 constant across all six: {all_tp_fp_constant}")
    print(f"TN+FN == 1521 constant across all six: {all_tn_fn_constant}")
    for name, res in matrix_checks.items():
        comp = res["computed_metrics"]
        rep = res["reported_metrics"]
        print(f"  {name:15}: acc={comp['accuracy']:.3f} (rep {rep['accuracy']}), prec={comp['precision']:.3f} (rep {rep['precision']}), rec={comp['recall']:.3f} (rep {rep['recall']})")
