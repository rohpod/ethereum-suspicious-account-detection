"""Pipeline B execution and leakage ladder (Rungs L1 and B) module.

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
- PLAN.md Section 8 (Global Convention) & Section 10 (Phase 2.0)
- docs/ASSUMPTIONS.md (Leakage-safe conventions)

Leakage Ladder Rungs:
- L1 ("SMOTE after split only"):
  1. Drop numeric NaNs on full raw data.
  2. Categorical token encoding and z-score scaling fit on ALL cleaned data (intentionally leaky).
  3. Load 22 PSO features from results/cache/pso_pipeline_a.json (strict cache hash verification).
  4. Stratified 80/20 train/test split (preserving natural class ratio in test set).
  5. SMOTE balancing applied to TRAIN split ONLY. Test set remains strictly natural (no synthetic rows).
  6. Fit and evaluate default models (XGBoost, SVM, IF, CART, LOF) on natural test set.

- B ("Full leakage-safe"):
  1. Drop numeric NaNs on raw data.
  2. Stratified 80/20 train/test split FIRST.
  3. TokenFrequencyEncoder and z-score scaler fit on TRAIN split only and applied to train and test.
  4. PSO feature selection run on training data only (with fold-wise SMOTE inside CV fitness).
  5. Subset training and test sets to PSO-selected features.
  6. SMOTE balancing applied to TRAIN split ONLY. Test set remains strictly natural.
  7. Fit and evaluate default models (XGBoost, SVM, IF, CART, LOF) on natural test set.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from imblearn import __version__ as imblearn_version
from xgboost import __version__ as xgboost_version

from src.data import (
    TokenFrequencyEncoder,
    load_raw,
    normalise_column_name,
    split_columns,
)
from src.evaluate import (
    check_consistency,
    compute_metrics,
    majority_baseline,
    write_results,
)
from src.models import (
    build_cart,
    build_isolation_forest,
    build_lof,
    build_svm,
    build_xgboost,
    fit_predict_scores,
)
from src.preprocess import (
    apply_scaler,
    apply_smote,
    clean_missing,
    fit_scaler,
    split_stratified,
)
from src.pso import (
    compare_with_table2,
    compute_pso_cache_hash,
    hash_index,
    run_pso,
)


def deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge source dictionary into target dictionary."""
    for k, v in source.items():
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            deep_merge(target[k], v)
        else:
            target[k] = copy.deepcopy(v)
    return target


def get_pipeline_b_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build Pipeline B configuration with overrides applied to a deep copy of cfg."""
    cfg_b = copy.deepcopy(cfg)
    overrides = cfg_b.get("pipeline_b", {})
    deep_merge(cfg_b, overrides)
    return cfg_b


def compare_with_pipeline_a(
    selected_b: list[str],
    selected_a: list[str],
) -> dict[str, Any]:
    """Compare Pipeline B selected features with Pipeline A selected features."""
    set_a = set(selected_a)
    set_b = set(selected_b)
    matched = sorted(list(set_a & set_b))
    missing_from_a = sorted(list(set_b - set_a))
    extra_in_a = sorted(list(set_a - set_b))
    union = set_a | set_b
    jaccard = float(len(matched) / len(union)) if union else 0.0

    return {
        "pipeline_a_count": len(selected_a),
        "pipeline_b_count": len(selected_b),
        "overlap_count": len(matched),
        "jaccard_similarity": jaccard,
        "matched_features": matched,
        "selected_in_b_not_in_a": missing_from_a,
        "selected_in_a_not_in_b": extra_in_a,
    }


def build_pipeline_l1_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build datasets for Leakage Ladder Rung L1 ('SMOTE after split only').

    Protocol:
    1. Clean missing numeric values on full data.
    2. Categorical encoding and z-score standardisation on ALL cleaned data (leaky).
    3. Load 22 PSO features from results/cache/pso_pipeline_a.json (verifying cache hash).
    4. Stratified 80/20 train/test split.
    5. SMOTE oversampling on TRAIN split only. Test split remains natural.

    Args:
        cfg: Configuration dictionary.

    Returns:
        dict containing train and test data, reports, and feature names.
    """
    # 1. Load raw data and split columns
    df, _ = load_raw(cfg)
    X_raw, y_raw, meta_raw = split_columns(df, cfg)

    # 2. Clean numeric NaNs
    X_clean, y_clean, meta_clean, clean_report = clean_missing(
        X_raw, y_raw, meta_raw, cfg
    )

    # 3. Categorical token encoding on full cleaned data (leaky)
    cat_cols = cfg.get("data", {}).get("categorical_columns", [])
    encoder = TokenFrequencyEncoder(columns=cat_cols)
    X_enc = encoder.fit_transform(X_clean)

    # 4. Standard scaling on full cleaned data (leaky)
    scaler = fit_scaler(X_enc)
    X_scaled = apply_scaler(scaler, X_enc)

    # 5. Load and verify Pipeline A PSO cache
    cache_path = (
        Path(cfg.get("paths", {}).get("results", "results/"))
        / "cache"
        / "pso_pipeline_a.json"
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Pipeline A PSO cache not found at {cache_path}. "
            "Pipeline A must be cached before running Rung L1."
        )

    with open(cache_path, "r", encoding="utf-8") as f:
        cached_data = json.load(f)

    seed = int(cfg.get("seed", 42))
    expected_hash = compute_pso_cache_hash(
        cfg,
        (len(X_scaled), len(X_scaled.columns)),
        list(X_scaled.columns),
        seed,
        pipeline="pipeline_a",
    )
    cached_hash = cached_data.get("hash")
    if cached_hash != expected_hash:
        raise ValueError(
            f"Pipeline A PSO cache hash mismatch ({cached_hash} != {expected_hash}). "
            "Cache corrupted or config modified."
        )

    selected_features = cached_data["result"]["selected_features"]

    # Verify no ID or label columns in features
    forbidden = {normalise_column_name(c) for c in cfg.get("data", {}).get("id_columns", [])}
    raw_label = cfg.get("data", {}).get("label_column")
    if raw_label:
        forbidden.add(normalise_column_name(raw_label))
    for feat in selected_features:
        if normalise_column_name(feat) in forbidden:
            raise ValueError(f"Forbidden column '{feat}' found in selected features.")

    X_subset = X_scaled[selected_features].copy()

    # 6. Stratified train/test split
    X_train_nat, X_test_nat, y_train_nat, y_test_nat = split_stratified(
        X_subset, y_clean, cfg
    )

    # 7. SMOTE on TRAIN split only
    X_train_res, y_train_res, is_syn_train, smote_report = apply_smote(
        X_train_nat, y_train_nat, cfg
    )

    # Test set is untouched and strictly natural
    is_syn_test = np.zeros(len(X_test_nat), dtype=bool)

    # Verify natural class ratio within 1 sample and zero synthetic rows in test
    clean_susp_ratio = clean_report["class_counts_after"][1] / clean_report["rows_after"]
    expected_test_susp = len(y_test_nat) * clean_susp_ratio
    actual_test_susp = int((y_test_nat == 1).sum())
    assert abs(actual_test_susp - expected_test_susp) < 1.0, (
        f"L1 test suspicious count {actual_test_susp} deviates by >= 1 from expected {expected_test_susp:.2f}"
    )
    assert np.sum(is_syn_test) == 0, "L1 test set contains synthetic rows."

    split_report = {
        "train_rows_natural": len(X_train_nat),
        "train_rows_resampled": len(X_train_res),
        "test_rows": len(X_test_nat),
        "train_class_counts_natural": {int(k): int(v) for k, v in y_train_nat.value_counts().items()},
        "train_class_counts_resampled": {int(k): int(v) for k, v in y_train_res.value_counts().items()},
        "test_class_counts": {int(k): int(v) for k, v in y_test_nat.value_counts().items()},
        "synthetic_train_count": int(np.sum(is_syn_train)),
        "synthetic_test_count": int(np.sum(is_syn_test)),
        "synthetic_test_ratio": 0.0,
        "natural_ratio_check": {
            "expected_test_suspicious": expected_test_susp,
            "actual_test_suspicious": actual_test_susp,
            "within_1_sample": bool(abs(actual_test_susp - expected_test_susp) < 1.0),
        },
    }

    return {
        "X_train": X_train_res,
        "y_train": y_train_res,
        "X_test": X_test_nat,
        "y_test": y_test_nat,
        "is_synthetic_train": is_syn_train,
        "is_synthetic_test": is_syn_test,
        "feature_names": selected_features,
        "smote_report": smote_report,
        "split_report": split_report,
        "clean_report": clean_report,
        "feature_set_name": "pso_pipeline_a_22",
    }


def build_pipeline_b_data(
    cfg: dict[str, Any],
    force_pso: bool = False,
) -> dict[str, Any]:
    """Build datasets for Full Leakage-Safe Pipeline B.

    Protocol:
    1. Drop numeric NaNs on raw data.
    2. Stratified 80/20 train/test split FIRST.
    3. Fit TokenFrequencyEncoder and z-score scaler on TRAIN split only; transform test.
    4. Run PSO on scaled train features (all 47) with smote_in_fold=True and train_index_hash.
    5. Subset features to PSO-selected features.
    6. SMOTE balancing applied to TRAIN split only. Test set remains strictly natural.

    Args:
        cfg: Configuration dictionary.
        force_pso: If True, bypass PSO cache and recompute.

    Returns:
        dict containing train and test data, PSO results, reports, and feature comparisons.
    """
    cfg_b = get_pipeline_b_cfg(cfg)

    # 1. Load raw data and split columns
    df, _ = load_raw(cfg_b)
    X_raw, y_raw, meta_raw = split_columns(df, cfg_b)

    # 2. Clean numeric NaNs
    X_clean, y_clean, meta_clean, clean_report = clean_missing(
        X_raw, y_raw, meta_raw, cfg_b
    )

    # 3. Stratified split FIRST
    X_train_raw, X_test_raw, y_train_nat, y_test_nat = split_stratified(
        X_clean, y_clean, cfg_b
    )

    # 4. Token frequency encoding fit on TRAIN only
    cat_cols = cfg_b.get("data", {}).get("categorical_columns", [])
    encoder = TokenFrequencyEncoder(columns=cat_cols)
    X_train_enc = encoder.fit_transform(X_train_raw)
    X_test_enc = encoder.transform(X_test_raw)

    # 5. Z-score scaler fit on TRAIN only
    scaler = fit_scaler(X_train_enc)
    X_train_scaled = apply_scaler(scaler, X_train_enc)
    X_test_scaled = apply_scaler(scaler, X_test_enc)

    # 6. Run PSO on scaled training data (all 47 features)
    train_idx_hash = hash_index(X_train_scaled.index)
    pso_result = run_pso(
        X_train_scaled,
        y_train_nat,
        cfg_b,
        force=force_pso,
        pipeline="pipeline_b",
        train_index_hash=train_idx_hash,
    )
    selected_features = pso_result["selected_features"]

    # Verify no ID or label columns in selected features
    forbidden = {normalise_column_name(c) for c in cfg_b.get("data", {}).get("id_columns", [])}
    raw_label = cfg_b.get("data", {}).get("label_column")
    if raw_label:
        forbidden.add(normalise_column_name(raw_label))
    for feat in selected_features:
        if normalise_column_name(feat) in forbidden:
            raise ValueError(f"Forbidden column '{feat}' found in selected features.")

    # 7. Subset to selected features
    X_train_sub = X_train_scaled[selected_features].copy()
    X_test_sub = X_test_scaled[selected_features].copy()

    # 8. SMOTE on TRAIN split only
    X_train_res, y_train_res, is_syn_train, smote_report = apply_smote(
        X_train_sub, y_train_nat, cfg_b
    )

    # Test set is untouched and natural
    is_syn_test = np.zeros(len(X_test_sub), dtype=bool)

    # Verify natural class ratio within 1 sample and zero synthetic rows in test
    clean_susp_ratio = clean_report["class_counts_after"][1] / clean_report["rows_after"]
    expected_test_susp = len(y_test_nat) * clean_susp_ratio
    actual_test_susp = int((y_test_nat == 1).sum())
    assert abs(actual_test_susp - expected_test_susp) < 1.0, (
        f"Pipeline B test suspicious count {actual_test_susp} deviates by >= 1 from expected {expected_test_susp:.2f}"
    )
    assert np.sum(is_syn_test) == 0, "Pipeline B test set contains synthetic rows."

    # Compare selected features with Pipeline A 22 features and Table 2 reference
    pso_a_cache_path = (
        Path(cfg_b.get("paths", {}).get("results", "results/"))
        / "cache"
        / "pso_pipeline_a.json"
    )
    if pso_a_cache_path.exists():
        with open(pso_a_cache_path, "r", encoding="utf-8") as f:
            pso_a_data = json.load(f)
        pipeline_a_features = pso_a_data.get("result", {}).get("selected_features", [])
        overlap_pipeline_a = compare_with_pipeline_a(selected_features, pipeline_a_features)
    else:
        overlap_pipeline_a = {"error": "Pipeline A PSO cache not available for comparison"}

    overlap_table2 = compare_with_table2(selected_features, cfg_b)

    split_report = {
        "train_rows_natural": len(X_train_sub),
        "train_rows_resampled": len(X_train_res),
        "test_rows": len(X_test_sub),
        "train_class_counts_natural": {int(k): int(v) for k, v in y_train_nat.value_counts().items()},
        "train_class_counts_resampled": {int(k): int(v) for k, v in y_train_res.value_counts().items()},
        "test_class_counts": {int(k): int(v) for k, v in y_test_nat.value_counts().items()},
        "synthetic_train_count": int(np.sum(is_syn_train)),
        "synthetic_test_count": int(np.sum(is_syn_test)),
        "synthetic_test_ratio": 0.0,
        "natural_ratio_check": {
            "expected_test_suspicious": expected_test_susp,
            "actual_test_suspicious": actual_test_susp,
            "within_1_sample": bool(abs(actual_test_susp - expected_test_susp) < 1.0),
        },
    }

    return {
        "X_train": X_train_res,
        "y_train": y_train_res,
        "X_test": X_test_sub,
        "y_test": y_test_nat,
        "is_synthetic_train": is_syn_train,
        "is_synthetic_test": is_syn_test,
        "feature_names": selected_features,
        "pso_result": {
            "best_rmse": pso_result["best_rmse"],
            "evaluations": pso_result["evaluations"],
            "n_selected": pso_result["n_selected"],
        },
        "overlap_pipeline_a": overlap_pipeline_a,
        "overlap_table2": overlap_table2,
        "smote_report": smote_report,
        "split_report": split_report,
        "clean_report": clean_report,
        "feature_set_name": "pso_pipeline_b",
        "cfg_b": cfg_b,
    }


def evaluate_rung_models(
    data: dict[str, Any],
    cfg: dict[str, Any],
    rung_name: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Fit default models and evaluate metrics on the natural test set.

    Models:
    - XGBoost (Table 5 defaults)
    - SVM (Table 6 defaults)
    - Isolation Forest (Table 7 defaults)
    - CART (Table 10 baseline)
    - LOF (Table 10 baseline)

    Args:
        data: Prepared dataset dictionary from build_pipeline_l1_data or build_pipeline_b_data.
        cfg: Configuration dictionary for this rung.
        rung_name: Identifier for stage timing prefix ('l1' or 'b').

    Returns:
        tuple of (rung_result_dict, stage_timings_dict).
    """
    stage_timings: dict[str, float] = {}
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    models_to_run = [
        ("xgboost", build_xgboost(cfg=cfg)),
        ("svm", build_svm(cfg=cfg)),
        ("isolation_forest", build_isolation_forest(cfg=cfg)),
        ("cart", build_cart(cfg=cfg)),
        ("lof", build_lof(cfg=cfg)),
    ]

    model_results: dict[str, Any] = {}

    for m_name, model in models_to_run:
        t_fit_start = time.time()
        y_pred, scores = fit_predict_scores(
            m_name, model, X_train, y_train, X_test
        )
        t_fit = time.time() - t_fit_start
        stage_timings[f"{rung_name}_{m_name}_fit_predict"] = t_fit

        metrics = compute_metrics(y_true=y_test, y_pred=y_pred, scores=scores)
        model_results[m_name] = {
            "metrics": metrics,
            "fit_predict_time_seconds": t_fit,
        }

    # Majority baseline on natural test split
    maj_base = majority_baseline(y_test)

    # Consistency checks across models
    expected_counts = {
        "n_test": len(y_test),
        0: int((y_test == 0).sum()),
        1: int((y_test == 1).sum()),
    }
    metrics_map = {m: res["metrics"] for m, res in model_results.items()}
    checks = check_consistency(metrics_map, expected_counts=expected_counts)

    rung_payload: dict[str, Any] = {
        "feature_count": len(data["feature_names"]),
        "features": data["feature_names"],
        "smote_report": data["smote_report"],
        "split_report": data["split_report"],
        "majority_baseline": maj_base,
        "models": model_results,
        "consistency_checks": checks,
    }

    if "pso_result" in data:
        rung_payload["pso_result"] = data["pso_result"]
        rung_payload["pso_overlap_with_pipeline_a"] = data["overlap_pipeline_a"]
        rung_payload["pso_overlap_with_table2"] = data["overlap_table2"]

    return rung_payload, stage_timings


def run_pipeline_b(
    cfg: dict[str, Any],
    rungs: list[str] = ["l1", "b"],
    force_pso: bool = False,
) -> dict[str, Any]:
    """Execute Pipeline B leakage ladder across specified rungs.

    Args:
        cfg: Base configuration dictionary.
        rungs: List of rungs to run ('l1', 'b', or both).
        force_pso: If True, bypass PSO cache and recompute.

    Returns:
        dict containing out_file path, payload, and stage timings.
    """
    t_start = time.time()
    stage_timings: dict[str, float] = {}
    rungs_results: dict[str, Any] = {}

    # 1. Rung L1 ("SMOTE after split only")
    if "l1" in rungs:
        t_l1_data_start = time.time()
        l1_data = build_pipeline_l1_data(cfg)
        t_l1_data = time.time() - t_l1_data_start
        stage_timings["l1_data_prep"] = t_l1_data

        l1_eval, l1_timings = evaluate_rung_models(l1_data, cfg, "l1")
        stage_timings.update(l1_timings)
        rungs_results["l1"] = l1_eval

    # 2. Rung B ("Full leakage-safe")
    if "b" in rungs:
        cfg_b = get_pipeline_b_cfg(cfg)
        t_b_data_start = time.time()
        b_data = build_pipeline_b_data(cfg, force_pso=force_pso)
        t_b_data = time.time() - t_b_data_start
        stage_timings["b_data_prep_and_pso"] = t_b_data

        b_eval, b_timings = evaluate_rung_models(b_data, cfg_b, "b")
        stage_timings.update(b_timings)
        rungs_results["b"] = b_eval

    total_time = time.time() - t_start
    stage_timings["total"] = total_time

    library_versions = {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost_version,
        "imbalanced-learn": imblearn_version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    payload = {
        "library_versions": library_versions,
        "paper_reference": cfg.get("paper_reference", {}),
        "rungs": rungs_results,
    }

    out_file = write_results(
        phase="phase2_0b",
        name="pipeline_b_ladder",
        payload=payload,
        cfg=cfg,
        timings=stage_timings,
    )

    return {
        "out_file": out_file,
        "payload": payload,
        "timings": stage_timings,
    }


def print_rung_summary(rung_name: str, rung_title: str, rung_data: dict[str, Any]) -> None:
    """Print formatted performance metrics table and reports for a given rung."""
    s_rep = rung_data["split_report"]
    maj = rung_data["majority_baseline"]
    f_count = rung_data["feature_count"]

    print(f"\n==========================================================================================")
    print(f"  RUNG {rung_name.upper()}: {rung_title} ({f_count} features)")
    print(f"==========================================================================================")
    print(
        f"Train rows (natural / resampled): {s_rep['train_rows_natural']} / {s_rep['train_rows_resampled']} | "
        f"Test rows: {s_rep['test_rows']}"
    )
    print(
        f"Test class balance: {s_rep['test_class_counts'][0]} Benign, {s_rep['test_class_counts'][1]} Suspicious "
        f"({s_rep['test_class_counts'][1]/s_rep['test_rows']*100:.2f}% suspicious)"
    )
    print(f"Synthetic test rows: {s_rep['synthetic_test_count']} / {s_rep['test_rows']} (0.00%)")
    print(
        f"Majority-Class Baseline: Always-Benign Accuracy = {maj['accuracy']:.4f} "
        f"({maj['accuracy']*100:.2f}%) | Suspicious Recall = {maj['recall_suspicious']:.4f}"
    )

    if "pso_result" in rung_data:
        pso_res = rung_data["pso_result"]
        print(f"\nPSO Feature Selection (Rung B):")
        print(f"  Selected Features: {pso_res['n_selected']} | Best RMSE: {pso_res['best_rmse']:.4f} | Evals: {pso_res['evaluations']}")
        if "pso_overlap_with_pipeline_a" in rung_data:
            ov_a = rung_data["pso_overlap_with_pipeline_a"]
            if "overlap_count" in ov_a:
                print(f"  Overlap with Pipeline A 22: {ov_a['overlap_count']}/22 (Jaccard: {ov_a['jaccard_similarity']:.4f})")
        if "pso_overlap_with_table2" in rung_data:
            ov_t2 = rung_data["pso_overlap_with_table2"]
            print(f"  Overlap with Table 2 (13 unambiguous): {ov_t2['overlap_count']}/13 (Jaccard: {ov_t2['jaccard_similarity']:.4f})")

    print("\nModel Evaluation on Natural Test Set:")
    header = (
        f"{'Model':<18} | {'Accuracy':<8} | {'Precision':<9} | {'Recall(Susp)':<12} | "
        f"{'F1':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'MCC':<8}"
    )
    print(header)
    print("-" * len(header))

    for m_name, m_info in rung_data["models"].items():
        met = m_info["metrics"]
        roc_str = f"{met['roc_auc']:.4f}" if met["roc_auc"] is not None else "N/A"
        pr_str = f"{met['pr_auc']:.4f}" if met["pr_auc"] is not None else "N/A"
        print(
            f"{m_name:<18} | {met['accuracy']:<8.4f} | {met['precision']:<9.4f} | "
            f"{met['recall']:<12.4f} | {met['f1']:<8.4f} | {roc_str:<8} | {pr_str:<8} | {met['mcc']:<8.4f}"
        )

    # Consistency checks summary
    all_passed = all(c["passed"] for c in rung_data["consistency_checks"])
    print(f"\nConsistency Checks: {'ALL PASSED' if all_passed else 'FAILED'}")
    for c in rung_data["consistency_checks"]:
        print(f"  - [{('PASS' if c['passed'] else 'FAIL')}] {c['name']}: {c['detail']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Pipeline B Leakage Ladder (Rungs L1 and B) with default models."
    )
    parser.add_argument(
        "--rung",
        choices=["l1", "b", "all"],
        default="all",
        help="Leakage ladder rung to execute (default: all).",
    )
    parser.add_argument(
        "--force-pso",
        action="store_true",
        help="Force recomputation of PSO for Rung B, bypassing existing cache.",
    )
    args = parser.parse_args()

    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rungs_to_run = ["l1", "b"] if args.rung == "all" else [args.rung]

    res = run_pipeline_b(cfg, rungs=rungs_to_run, force_pso=args.force_pso)
    payload = res["payload"]
    results = payload["rungs"]
    out_file = res["out_file"]
    timings = res["timings"]

    print("\n==========================================================================================")
    print(f"  PIPELINE B LEAKAGE LADDER COMPLETED")
    print(f"  Output saved to: {out_file}")
    print(f"==========================================================================================")

    rung_titles = {
        "l1": "SMOTE after split only (leaky scaling & Pipeline A 22 features)",
        "b": "Full Leakage-Safe (stratified split first, train-only scaling & PSO)",
    }

    for r_name in rungs_to_run:
        if r_name in results:
            print_rung_summary(r_name, rung_titles[r_name], results[r_name])

    print("\nExecution Timings (seconds):")
    for k, v in timings.items():
        print(f"  {k:<30}: {v:.2f}s")
