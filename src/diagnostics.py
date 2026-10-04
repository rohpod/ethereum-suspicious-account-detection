"""Diagnostic analysis module for Ethereum suspicious account detection.

Reference:
- PLAN.md Phase 2.0 (Task 2.0c Part 1: Duplicate Diagnostics)
- docs/ASSUMPTIONS.md (Duplicate feature vectors and addresses retention policy)

Provides:
- Exact duplicate feature detection overall and per class on the cleaned, unscaled 47-feature matrix.
- Contradictory group detection (identical feature vectors with conflicting labels).
- Train/test leakage overlap detection (test rows whose exact feature vector appears in train split).
- XGBoost evaluation comparing full natural test set vs. strictly unseen test rows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data import (
    TokenFrequencyEncoder,
    load_raw,
    split_columns,
)
from src.evaluate import (
    compute_metrics,
    majority_baseline,
    write_results,
)
from src.models import build_xgboost, fit_predict_scores
from src.pipeline_b import (
    build_pipeline_b_data,
    get_pipeline_b_cfg,
)
from src.preprocess import clean_missing, split_stratified


def compute_duplicate_diagnostics(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute detailed duplicate, contradiction, and train/test leakage statistics.

    Args:
        X_all: Full cleaned and encoded feature matrix (unscaled).
        y_all: Full target labels Series.
        X_train: Training feature matrix (unscaled).
        y_train: Training target labels Series.
        X_test: Testing feature matrix (unscaled).
        y_test: Testing target labels Series.

    Returns:
        Dictionary containing:
        - duplicate_rows: Counts of duplicate rows (keep='first' and keep=False) overall and per class.
        - contradictory_groups: Count of identical-feature clusters with conflicting labels.
        - train_test_overlap: Counts of test rows appearing in train, and unseen test rows.
        - unseen_mask: Boolean array marking unseen test rows.
    """
    # (a) Exact duplicate feature rows overall and per class
    # keep='first' counts redundant rows beyond the first instance
    dup_redundant_mask = X_all.duplicated(keep="first")
    # keep=False marks all rows that belong to any duplicate group
    dup_cluster_mask = X_all.duplicated(keep=False)

    total_rows = len(X_all)
    dup_redundant_count = int(dup_redundant_mask.sum())
    dup_cluster_count = int(dup_cluster_mask.sum())

    dup_redundant_by_class = {
        int(k): int(v) for k, v in y_all[dup_redundant_mask].value_counts().items()
    }
    dup_redundant_by_class.setdefault(0, 0)
    dup_redundant_by_class.setdefault(1, 0)

    dup_cluster_by_class = {
        int(k): int(v) for k, v in y_all[dup_cluster_mask].value_counts().items()
    }
    dup_cluster_by_class.setdefault(0, 0)
    dup_cluster_by_class.setdefault(1, 0)

    # (b) Contradictory groups (identical feature vectors, different labels)
    # Convert features to tuple representation for exact hash grouping
    feature_tuples_all = [tuple(row) for row in X_all.values]
    grouped_labels: dict[tuple[Any, ...], list[int]] = {}
    for feat_tup, label in zip(feature_tuples_all, y_all):
        grouped_labels.setdefault(feat_tup, []).append(int(label))

    contradictory_groups_count = 0
    contradictory_rows_count = 0
    contradictory_samples: list[dict[str, Any]] = []

    for feat_tup, labels in grouped_labels.items():
        if len(set(labels)) > 1:
            contradictory_groups_count += 1
            contradictory_rows_count += len(labels)
            if len(contradictory_samples) < 5:
                contradictory_samples.append({
                    "cluster_size": len(labels),
                    "label_counts": {
                        0: labels.count(0),
                        1: labels.count(1),
                    },
                })

    # (c) Test rows appearing in train vs. unseen test rows
    train_feature_set = {tuple(row) for row in X_train.values}
    test_in_train = [tuple(row) in train_feature_set for row in X_test.values]
    test_in_train_arr = np.array(test_in_train, dtype=bool)
    unseen_test_arr = ~test_in_train_arr

    overlap_count = int(test_in_train_arr.sum())
    overlap_by_class = {
        int(k): int(v) for k, v in y_test[test_in_train_arr].value_counts().items()
    }
    overlap_by_class.setdefault(0, 0)
    overlap_by_class.setdefault(1, 0)

    unseen_count = int(unseen_test_arr.sum())
    unseen_by_class = {
        int(k): int(v) for k, v in y_test[unseen_test_arr].value_counts().items()
    }
    unseen_by_class.setdefault(0, 0)
    unseen_by_class.setdefault(1, 0)

    return {
        "duplicate_rows": {
            "total_rows": total_rows,
            "redundant_rows_count": dup_redundant_count,
            "redundant_ratio": float(dup_redundant_count / total_rows) if total_rows > 0 else 0.0,
            "redundant_by_class": dup_redundant_by_class,
            "cluster_rows_count": dup_cluster_count,
            "cluster_ratio": float(dup_cluster_count / total_rows) if total_rows > 0 else 0.0,
            "cluster_by_class": dup_cluster_by_class,
        },
        "contradictory_groups": {
            "groups_count": contradictory_groups_count,
            "rows_count": contradictory_rows_count,
            "samples": contradictory_samples,
        },
        "train_test_overlap": {
            "test_total": len(X_test),
            "test_rows_in_train_count": overlap_count,
            "test_rows_in_train_ratio": float(overlap_count / len(X_test)) if len(X_test) > 0 else 0.0,
            "test_rows_in_train_by_class": overlap_by_class,
            "unseen_test_count": unseen_count,
            "unseen_test_ratio": float(unseen_count / len(X_test)) if len(X_test) > 0 else 0.0,
            "unseen_test_by_class": unseen_by_class,
        },
        "unseen_mask": unseen_test_arr,
    }


def run_duplicate_diagnostic(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute complete duplicate diagnostic and XGBoost evaluation.

    Args:
        cfg: Base configuration dictionary.

    Returns:
        dict containing out_file path, diagnostic reports, and performance metrics.
    """
    t_start = time.time()
    cfg_b = get_pipeline_b_cfg(cfg)

    # 1. Load raw data and clean numeric NaNs
    df, _ = load_raw(cfg_b)
    X_raw, y_raw, meta_raw = split_columns(df, cfg_b)
    X_clean, y_clean, meta_clean, clean_report = clean_missing(
        X_raw, y_raw, meta_raw, cfg_b
    )

    # 2. Split first using rung B semantics
    X_train_raw, X_test_raw, y_train_nat, y_test_nat = split_stratified(
        X_clean, y_clean, cfg_b
    )

    # 3. Fit TokenFrequencyEncoder on train only, transform both
    cat_cols = cfg_b.get("data", {}).get("categorical_columns", [])
    encoder = TokenFrequencyEncoder(columns=cat_cols)
    X_train_enc = encoder.fit_transform(X_train_raw)
    X_test_enc = encoder.transform(X_test_raw)

    # Combine back to form the full cleaned, unscaled 47-feature matrix
    X_all_enc = pd.concat([X_train_enc, X_test_enc]).loc[X_clean.index]
    y_all = y_clean.copy()

    # 4. Compute duplicate and contradiction diagnostics on the 47-feature matrix
    t_diag_start = time.time()
    diag_res = compute_duplicate_diagnostics(
        X_all_enc, y_all, X_train_enc, y_train_nat, X_test_enc, y_test_nat
    )
    t_diag = time.time() - t_diag_start
    unseen_mask = diag_res["unseen_mask"]

    # 5. Build rung B dataset (scaled, PSO features selected, SMOTE'd train)
    t_prep_start = time.time()
    b_data = build_pipeline_b_data(cfg, force_pso=False)
    t_prep = time.time() - t_prep_start

    X_train_res = b_data["X_train"]
    y_train_res = b_data["y_train"]
    X_test_b = b_data["X_test"]
    y_test_b = b_data["y_test"]

    # 6. Fit default XGBoost on rung B's SMOTE'd train
    t_fit_start = time.time()
    xgb_model = build_xgboost(cfg=cfg_b)
    y_pred, scores = fit_predict_scores(
        "xgboost", xgb_model, X_train_res, y_train_res, X_test_b
    )
    t_fit = time.time() - t_fit_start

    # (i) Full natural test set metrics (assert matches 2.0b within 1e-9)
    full_metrics = compute_metrics(y_true=y_test_b, y_pred=y_pred, scores=scores)
    full_maj_baseline = majority_baseline(y_test_b)

    # Load 2.0b results to verify reproduction
    results_dir = Path(cfg.get("paths", {}).get("results", "results/"))
    matches_20b = sorted(results_dir.glob("phase2_0b_pipeline_b_ladder_*.json"))
    if matches_20b:
        with open(matches_20b[-1], "r", encoding="utf-8") as f:
            res_20b = json.load(f)
        xgb_20b = res_20b.get("rungs", {}).get("b", {}).get("models", {}).get("xgboost", {}).get("metrics", {})
        if xgb_20b:
            for metric_k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]:
                diff = abs(full_metrics[metric_k] - xgb_20b[metric_k])
                assert diff < 1e-9, (
                    f"Reproduction mismatch for {metric_k}: {full_metrics[metric_k]} != {xgb_20b[metric_k]} (diff={diff})"
                )

    # (ii) Unseen test rows metrics and majority baseline
    y_test_unseen = y_test_b[unseen_mask]
    y_pred_unseen = y_pred[unseen_mask]
    scores_unseen = scores[unseen_mask]

    unseen_metrics = compute_metrics(
        y_true=y_test_unseen, y_pred=y_pred_unseen, scores=scores_unseen
    )
    unseen_maj_baseline = majority_baseline(y_test_unseen)

    total_time = time.time() - t_start
    timings = {
        "diagnostics_analysis": t_diag,
        "data_prep": t_prep,
        "xgboost_fit_predict": t_fit,
        "total": total_time,
    }

    payload = {
        "diagnostic_summary": {
            "duplicate_rows": diag_res["duplicate_rows"],
            "contradictory_groups": diag_res["contradictory_groups"],
            "train_test_overlap": diag_res["train_test_overlap"],
        },
        "full_test_evaluation": {
            "test_rows": len(y_test_b),
            "class_counts": {int(k): int(v) for k, v in y_test_b.value_counts().items()},
            "majority_baseline": full_maj_baseline,
            "metrics": full_metrics,
        },
        "unseen_test_evaluation": {
            "test_rows": int(unseen_mask.sum()),
            "class_counts": {int(k): int(v) for k, v in y_test_unseen.value_counts().items()},
            "majority_baseline": unseen_maj_baseline,
            "metrics": unseen_metrics,
        },
        "deltas_full_vs_unseen": {
            k: float(unseen_metrics[k] - full_metrics[k])
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
        },
    }

    out_file = write_results(
        phase="phase2_0c",
        name="duplicates",
        payload=payload,
        cfg=cfg,
        timings=timings,
    )

    return {
        "out_file": out_file,
        "payload": payload,
        "timings": timings,
    }


def print_diagnostic_summary(payload: dict[str, Any]) -> None:
    """Print readable console summary of duplicate diagnostics."""
    diag = payload["diagnostic_summary"]
    dups = diag["duplicate_rows"]
    contra = diag["contradictory_groups"]
    overlap = diag["train_test_overlap"]
    full_eval = payload["full_test_evaluation"]
    unseen_eval = payload["unseen_test_evaluation"]

    print("\n==========================================================================================")
    print("  DUPLICATE & TRAIN/TEST LEAKAGE DIAGNOSTIC SUMMARY")
    print("==========================================================================================")
    print(f"(a) Exact Duplicate Feature Rows (Cleaned 47-Feature Matrix, N={dups['total_rows']}):")
    print(
        f"    - Redundant rows (keep='first'): {dups['redundant_rows_count']} "
        f"({dups['redundant_ratio']*100:.2f}% of dataset)"
    )
    print(f"      By class: Benign (0) = {dups['redundant_by_class'][0]}, Suspicious (1) = {dups['redundant_by_class'][1]}")
    print(
        f"    - Cluster rows (keep=False):   {dups['cluster_rows_count']} "
        f"({dups['cluster_ratio']*100:.2f}% of dataset)"
    )
    print(f"      By class: Benign (0) = {dups['cluster_by_class'][0]}, Suspicious (1) = {dups['cluster_by_class'][1]}")

    print(f"\n(b) Contradictory Groups (Identical features, Different labels):")
    print(f"    - Contradictory groups found: {contra['groups_count']}")
    print(f"    - Total rows involved:        {contra['rows_count']}")

    print(f"\n(c) Train/Test Feature Overlap (Stratified Split, Test N={overlap['test_total']}):")
    print(
        f"    - Test rows appearing in train: {overlap['test_rows_in_train_count']} "
        f"({overlap['test_rows_in_train_ratio']*100:.2f}% of test set)"
    )
    print(f"      By class: Benign (0) = {overlap['test_rows_in_train_by_class'][0]}, Suspicious (1) = {overlap['test_rows_in_train_by_class'][1]}")
    print(
        f"    - Unseen test rows:             {overlap['unseen_test_count']} "
        f"({overlap['unseen_test_ratio']*100:.2f}% of test set)"
    )
    print(f"      By class: Benign (0) = {overlap['unseen_test_by_class'][0]}, Suspicious (1) = {overlap['unseen_test_by_class'][1]}")

    print("\n==========================================================================================")
    print("  XGBOOST TEST PERFORMANCE: FULL TEST SET vs. UNSEEN TEST ROWS")
    print("==========================================================================================")
    header = (
        f"{'Evaluation Subset':<22} | {'N_test':<6} | {'Majority':<8} | {'Accuracy':<8} | "
        f"{'Precision':<9} | {'Recall(Susp)':<12} | {'F1':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'MCC':<8}"
    )
    print(header)
    print("-" * len(header))

    fm = full_eval["metrics"]
    fmaj = full_eval["majority_baseline"]
    print(
        f"{'Full Natural Test':<22} | {full_eval['test_rows']:<6} | {fmaj['accuracy']:<8.4f} | "
        f"{fm['accuracy']:<8.4f} | {fm['precision']:<9.4f} | {fm['recall']:<12.4f} | "
        f"{fm['f1']:<8.4f} | {fm['roc_auc']:<8.4f} | {fm['pr_auc']:<8.4f} | {fm['mcc']:<8.4f}"
    )

    um = unseen_eval["metrics"]
    umaj = unseen_eval["majority_baseline"]
    print(
        f"{'Unseen Test Rows Only':<22} | {unseen_eval['test_rows']:<6} | {umaj['accuracy']:<8.4f} | "
        f"{um['accuracy']:<8.4f} | {um['precision']:<9.4f} | {um['recall']:<12.4f} | "
        f"{um['f1']:<8.4f} | {um['roc_auc']:<8.4f} | {um['pr_auc']:<8.4f} | {um['mcc']:<8.4f}"
    )

    deltas = payload["deltas_full_vs_unseen"]
    print(f"\nDelta (Unseen - Full):")
    print(
        f"  Acc: {deltas['accuracy']:+.4f} | Prec: {deltas['precision']:+.4f} | "
        f"Rec: {deltas['recall']:+.4f} | F1: {deltas['f1']:+.4f} | "
        f"ROC-AUC: {deltas['roc_auc']:+.4f} | PR-AUC: {deltas['pr_auc']:+.4f} | MCC: {deltas['mcc']:+.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostics for Ethereum account detection.")
    parser.add_argument(
        "--duplicates",
        action="store_true",
        help="Run duplicate, contradiction, and train/test leakage diagnostic.",
    )
    args = parser.parse_args()

    if not args.duplicates:
        parser.print_help()
        exit(0)

    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    res = run_duplicate_diagnostic(cfg)
    print_diagnostic_summary(res["payload"])
    print(f"\nDiagnostic results written to: {res['out_file']}")
