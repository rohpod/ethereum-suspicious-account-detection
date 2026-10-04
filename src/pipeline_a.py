"""Pipeline A execution and default model baselines module.

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
  Section 4.2.4 (Data Balancing: SMOTE)
  Section 4.2.5 (Data Splitting: 80% train, 20% test)
  Section 5 & Tables 8-9 (Baseline and GA confusion matrices & metrics)
  Algorithm 6 (Full Pipeline A order: clean -> z-score -> PSO -> SMOTE -> split -> models)

Pipeline A Execution Order:
1. Data cleaning (dropping numeric NaNs: 829 rows dropped -> 9,012 accounts).
2. Categorical token frequency encoding.
3. Z-score standard scaling on all 9,012 accounts.
4. Feature selection (PSO selected features loaded from verified cache, or Table 2 reference).
5. SMOTE balancing applied to the entire feature matrix BEFORE splitting (7,662 -> 15,324 accounts).
6. Unstratified 80/20 train/test split (12,259 train / 3,065 test).
7. Default model fitting & test set evaluation (XGBoost, SVM, Isolation Forest).
"""

from __future__ import annotations

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
from src.evaluate import check_consistency, compute_metrics, write_results
from src.models import (
    build_isolation_forest,
    build_svm,
    build_xgboost,
    fit_predict_scores,
)
from src.preprocess import (
    apply_scaler,
    apply_smote,
    clean_missing,
    fit_scaler,
    split_train_test,
)
from src.pso import compute_pso_cache_hash


def build_pipeline_a_data(
    cfg: dict[str, Any],
    feature_set: str = "pso",
) -> dict[str, Any]:
    """Build deterministic train and test datasets for Pipeline A.

    Executes data preparation following paper Algorithm 6 Steps 1-3:
    Clean -> Encode -> Z-score -> Feature Select -> SMOTE -> 80/20 Split.

    Args:
        cfg: Configuration dictionary.
        feature_set: Name of feature set to extract ('pso' or 'table2').

    Returns:
        dict containing:
            - X_train: Training feature DataFrame (12,259 rows).
            - X_test: Testing feature DataFrame (3,065 rows).
            - y_train: Training binary label Series.
            - y_test: Testing binary label Series.
            - is_synthetic_train: Boolean array marking synthetic training rows.
            - is_synthetic_test: Boolean array marking synthetic test rows.
            - feature_names: List of selected feature column names.
            - smote_report: Report dict from apply_smote.
            - split_report: Report dict from split_train_test.
            - feature_set_name: Name of the feature set used.

    Raises:
        FileNotFoundError: If PSO cache is missing when feature_set='pso'.
        ValueError: If cache hash mismatches or forbidden ID columns are detected.
    """
    # 1. Load raw data and split columns
    df, _ = load_raw(cfg)
    X_raw, y_raw, meta_raw = split_columns(df, cfg)

    # 2. Clean missing numeric values
    X_clean, y_clean, _meta_clean, clean_report = clean_missing(
        X_raw, y_raw, meta_raw, cfg
    )

    # 3. Categorical token encoding
    cat_cols = cfg.get("data", {}).get("categorical_columns", [])
    encoder = TokenFrequencyEncoder(columns=cat_cols)
    X_enc = encoder.fit_transform(X_clean)

    # 4. Fit and apply z-score scaler
    scaler = fit_scaler(X_enc)
    X_scaled = apply_scaler(scaler, X_enc)

    # 5. Resolve feature set
    norm_set = feature_set.strip().lower()
    if norm_set == "pso":
        cache_path = (
            Path(cfg.get("paths", {}).get("results", "results/"))
            / "cache"
            / "pso_pipeline_a.json"
        )
        if not cache_path.exists():
            raise FileNotFoundError(
                f"PSO cache not found at {cache_path}. "
                "Phase 1.4 cache must exist before running Pipeline A."
            )

        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)

        seed = int(cfg.get("seed", 42))
        expected_hash = compute_pso_cache_hash(
            cfg,
            (len(X_scaled), len(X_scaled.columns)),
            list(X_scaled.columns),
            seed,
        )

        cached_hash = cached_data.get("hash")
        if cached_hash != expected_hash:
            raise ValueError(
                f"PSO cache hash mismatch ({cached_hash} != {expected_hash}). "
                "Config or features have changed. Recomputing PSO is prohibited in Phase 1.5."
            )

        selected_features = cached_data["result"]["selected_features"]

    elif norm_set in {"table2", "table2_reference"}:
        selected_features = cfg.get("features", {}).get("table2_reference", [])
        if not selected_features:
            raise ValueError(
                "Config missing 'features.table2_reference' feature list."
            )
    else:
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. Expected 'pso' or 'table2'."
        )

    # Verify no ID or label columns are included
    forbidden = {normalise_column_name(c) for c in cfg.get("data", {}).get("id_columns", [])}
    raw_label = cfg.get("data", {}).get("label_column")
    if raw_label:
        forbidden.add(normalise_column_name(raw_label))

    for feat in selected_features:
        if normalise_column_name(feat) in forbidden:
            raise ValueError(
                f"Forbidden identifier or label column '{feat}' in feature set '{feature_set}'."
            )

    X_subset = X_scaled[selected_features].copy()

    # 6. Apply SMOTE to the selected feature matrix on full dataset
    X_res, y_res, is_synthetic, smote_report = apply_smote(
        X_subset, y_clean, cfg
    )

    # 7. Unstratified 80/20 random split
    (
        X_train,
        X_test,
        y_train,
        y_test,
        is_syn_train,
        is_syn_test,
        split_report,
    ) = split_train_test(X_res, y_res, is_synthetic, cfg)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "is_synthetic_train": is_syn_train,
        "is_synthetic_test": is_syn_test,
        "feature_names": selected_features,
        "smote_report": smote_report,
        "split_report": split_report,
        "clean_report": clean_report,
        "feature_set_name": norm_set,
    }


def run_pipeline_a_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute Pipeline A benchmark across PSO and Table 2 feature sets.

    Trains default XGBoost, SVM, and Isolation Forest models on both
    feature sets, evaluates test metrics, and calculates deltas to paper Table 8.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Full experiment result dictionary.
    """
    t_start = time.time()
    feature_sets = ["pso", "table2"]
    experiment_results: dict[str, Any] = {}
    stage_timings: dict[str, float] = {}

    paper_defaults = cfg.get("paper_results", {}).get("default", {})

    for fset_name in feature_sets:
        t_data_start = time.time()
        data = build_pipeline_a_data(cfg, feature_set=fset_name)
        t_data = time.time() - t_data_start
        stage_timings[f"{fset_name}_data_prep"] = t_data

        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

        models_to_run = [
            ("xgboost", build_xgboost(cfg=cfg)),
            ("svm", build_svm(cfg=cfg)),
            ("isolation_forest", build_isolation_forest(cfg=cfg)),
        ]

        fset_model_results: dict[str, Any] = {}

        for m_name, model in models_to_run:
            t_fit_start = time.time()
            y_pred, scores = fit_predict_scores(
                m_name, model, X_train, y_train, X_test
            )
            t_fit = time.time() - t_fit_start
            stage_timings[f"{fset_name}_{m_name}_fit_predict"] = t_fit

            metrics = compute_metrics(y_true=y_test, y_pred=y_pred, scores=scores)

            # Compute deltas to paper reported values
            deltas: dict[str, float] = {}
            paper_model = paper_defaults.get(m_name, {})
            if paper_model:
                for k in ["accuracy", "precision", "recall", "f1"]:
                    if k in paper_model:
                        deltas[k] = float(metrics[k] - paper_model[k])
                if "auc" in paper_model and metrics["roc_auc"] is not None:
                    deltas["auc"] = float(metrics["roc_auc"] - paper_model["auc"])

            fset_model_results[m_name] = {
                "metrics": metrics,
                "deltas_to_paper": deltas,
                "fit_predict_time_seconds": t_fit,
            }

        # Run consistency check across the 3 models for this feature set
        metrics_for_check = {m: res["metrics"] for m, res in fset_model_results.items()}
        checks = check_consistency(metrics_for_check)

        experiment_results[fset_name] = {
            "feature_count": len(data["feature_names"]),
            "features": data["feature_names"],
            "smote_report": data["smote_report"],
            "split_report": data["split_report"],
            "models": fset_model_results,
            "consistency_checks": checks,
        }

    total_time = time.time() - t_start
    stage_timings["total"] = total_time

    # Library versions
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
        "results": experiment_results,
    }

    out_file = write_results(
        phase="phase1_5",
        name="pipeline_a_defaults",
        payload=payload,
        cfg=cfg,
        timings=stage_timings,
    )

    return {
        "out_file": out_file,
        "payload": payload,
        "timings": stage_timings,
    }


if __name__ == "__main__":
    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    res = run_pipeline_a_defaults(cfg)
    payload = res["payload"]
    results = payload["results"]
    out_file = res["out_file"]

    print("\n=== Pipeline A Default Models Benchmark Completed ===")
    print(f"Results written to: {out_file}")

    for fset in ["pso", "table2"]:
        data_res = results[fset]
        s_rep = data_res["split_report"]
        print(f"\n--- Feature Set: {fset.upper()} ({data_res['feature_count']} features) ---")
        print(f"Train rows: {s_rep['train_rows']} | Test rows: {s_rep['test_rows']}")
        print(f"Test class balance: {s_rep['test_class_counts']} (Benign vs Suspicious)")
        print(
            f"Test synthetic rows: {s_rep['synthetic_test_count']} / {s_rep['test_rows']} "
            f"({s_rep['synthetic_test_ratio']*100:.1f}%)"
        )
        print("Model Performance on Test Set:")
        for m_name, m_res in data_res["models"].items():
            met = m_res["metrics"]
            deltas = m_res["deltas_to_paper"]
            auc_val = f"{met['roc_auc']:.3f}" if met["roc_auc"] is not None else "N/A"
            print(
                f"  {m_name:17}: Acc={met['accuracy']:.4f} (Δ {deltas.get('accuracy', 0.0):+.4f}) | "
                f"F1={met['f1']:.4f} (Δ {deltas.get('f1', 0.0):+.4f}) | "
                f"Prec={met['precision']:.4f} | Rec={met['recall']:.4f} | AUC={auc_val}"
            )
