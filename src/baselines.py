"""Comparison baselines execution module (CART and LOF, Table 10).

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
  Section 5 (Experimental Results)
  Table 10 (Comparison with existing methods on Ethereum dataset):
    - Saxena et al. [14]: CART
    - Fangfang et al. [12]: LOF
    - Alarab et al. [16]: XGBoost (default baseline)
    - El-Attar et al. [ours]: Proposed framework (XGBoost with GA)
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

from src.evaluate import check_consistency, compute_metrics, write_results
from src.models import build_cart, build_lof, fit_predict_scores
from src.pipeline_a import build_pipeline_a_data


def find_latest_result(results_dir: Path, pattern: str) -> Path:
    """Find the most recent results JSON matching the specified pattern.

    Args:
        results_dir: Path to directory containing results.
        pattern: Glob pattern to match.

    Returns:
        Path to latest matching file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No result files matching pattern '{pattern}' found in {results_dir}"
        )
    return matches[-1]


def compute_deltas_to_table10(
    metrics: dict[str, Any],
    paper_row: dict[str, Any],
) -> dict[str, float]:
    """Compute difference between evaluated metrics and Table 10 reported values.

    Delta = metric_evaluated - metric_paper.

    Args:
        metrics: Computed metrics dictionary.
        paper_row: Table 10 reference row dictionary from config.

    Returns:
        dict mapping metric name to delta float.
    """
    deltas: dict[str, float] = {}
    scalar_keys = ["accuracy", "mae", "precision", "recall", "f1"]
    for k in scalar_keys:
        if k in paper_row and metrics.get(k) is not None:
            deltas[k] = float(metrics[k] - paper_row[k])

    if "auc" in paper_row and metrics.get("roc_auc") is not None:
        deltas["auc"] = float(metrics["roc_auc"] - paper_row["auc"])

    return deltas


def run_baselines(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train and evaluate CART and LOF baselines on Pipeline A test data.

    Evaluates across both 'pso' (22 features) and 'table2' (14 features) sets.
    Reuses default XGBoost metrics from Phase 1.5 without retraining.
    Computes deltas against paper Table 10 values and runs consistency checks.

    Args:
        cfg: Configuration dictionary.

    Returns:
        dict containing out_file path, payload, and execution timings.
    """
    t_start = time.time()
    stage_timings: dict[str, float] = {}

    results_dir = Path(cfg.get("paths", {}).get("results", "results/"))

    # 1. Load Phase 1.5 defaults to reuse default XGBoost metrics
    phase1_5_file = find_latest_result(
        results_dir, "phase1_5_pipeline_a_defaults_*.json"
    )
    with open(phase1_5_file, "r", encoding="utf-8") as f:
        phase1_5_data = json.load(f)

    phase1_5_fsets = phase1_5_data.get("results", {}).get(
        "feature_sets", phase1_5_data.get("results", {})
    )

    table10_cfg = cfg.get("paper_results", {}).get("table10", {})
    cart_paper = table10_cfg.get("cart", {})
    lof_paper = table10_cfg.get("lof", {})
    xgb_default_paper = table10_cfg.get("xgboost_default", {})

    experiment_results: dict[str, Any] = {}

    for fset_name in ["pso", "table2"]:
        t0_fset = time.time()

        # Build Pipeline A data for this feature set
        data = build_pipeline_a_data(cfg, feature_set=fset_name)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Extract Phase 1.5 XGBoost default metrics
        xgb_default_entry = (
            phase1_5_fsets.get(fset_name, {})
            .get("models", {})
            .get("xgboost", {})
        )
        if not xgb_default_entry:
            raise KeyError(
                f"Missing xgboost default results for '{fset_name}' in {phase1_5_file}"
            )
        xgb_metrics = xgb_default_entry["metrics"]
        xgb_deltas = compute_deltas_to_table10(xgb_metrics, xgb_default_paper)

        # Evaluate CART
        t0_cart = time.time()
        cart_model = build_cart(cfg=cfg)
        y_pred_cart, scores_cart = fit_predict_scores(
            "cart", cart_model, X_train, y_train, X_test
        )
        metrics_cart = compute_metrics(y_test, y_pred_cart, scores_cart)
        deltas_cart = compute_deltas_to_table10(metrics_cart, cart_paper)
        t_cart = time.time() - t0_cart
        stage_timings[f"{fset_name}_cart"] = t_cart

        # Evaluate LOF (unsupervised novelty detector)
        t0_lof = time.time()
        lof_model = build_lof(cfg=cfg)
        y_pred_lof, scores_lof = fit_predict_scores(
            "lof", lof_model, X_train, y_train, X_test
        )
        metrics_lof = compute_metrics(y_test, y_pred_lof, scores_lof)
        deltas_lof = compute_deltas_to_table10(metrics_lof, lof_paper)
        t_lof = time.time() - t0_lof
        stage_timings[f"{fset_name}_lof"] = t_lof

        # Run consistency checks across all 3 models on this test set
        metrics_for_check = {
            "cart": metrics_cart,
            "lof": metrics_lof,
            "xgboost_default": xgb_metrics,
        }
        checks = check_consistency(
            metrics_for_check,
            expected_counts={"n_test": len(y_test)},
        )

        fset_model_results = {
            "cart": {
                "algorithm": "CART",
                "reference": cart_paper.get("reference", "[14]"),
                "method_author": cart_paper.get("method", "Saxena et al."),
                "hyperparameters": {
                    "criterion": cart_model.criterion,
                    "max_depth": cart_model.max_depth,
                    "min_samples_split": cart_model.min_samples_split,
                    "min_samples_leaf": cart_model.min_samples_leaf,
                    "random_state": cart_model.random_state,
                },
                "metrics": metrics_cart,
                "deltas_to_paper_table10": deltas_cart,
                "paper_table10_reference": cart_paper,
                "fit_time_seconds": t_cart,
            },
            "lof": {
                "algorithm": "LOF",
                "reference": lof_paper.get("reference", "[12]"),
                "method_author": lof_paper.get("method", "Fangfang et al."),
                "hyperparameters": {
                    "n_neighbors": lof_model.n_neighbors,
                    "contamination": lof_model.contamination,
                    "novelty": lof_model.novelty,
                },
                "metrics": metrics_lof,
                "deltas_to_paper_table10": deltas_lof,
                "paper_table10_reference": lof_paper,
                "fit_time_seconds": t_lof,
            },
            "xgboost_default": {
                "algorithm": "XGBoost",
                "reference": xgb_default_paper.get("reference", "[16]"),
                "method_author": xgb_default_paper.get("method", "Alarab et al."),
                "source_file": str(phase1_5_file),
                "metrics": xgb_metrics,
                "deltas_to_paper_table10": xgb_deltas,
                "paper_table10_reference": xgb_default_paper,
                "reused_from_phase1_5": True,
            },
        }

        experiment_results[fset_name] = {
            "feature_count": len(data["feature_names"]),
            "features": data["feature_names"],
            "smote_report": data["smote_report"],
            "split_report": data["split_report"],
            "models": fset_model_results,
            "consistency_checks": checks,
        }
        stage_timings[f"{fset_name}_total"] = time.time() - t0_fset

    total_time = time.time() - t_start
    stage_timings["total"] = total_time

    library_versions = {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    payload = {
        "source_phase1_5_file": str(phase1_5_file),
        "library_versions": library_versions,
        "paper_table10": table10_cfg,
        "feature_sets": experiment_results,
        "results": experiment_results,
    }

    out_file = write_results(
        phase="phase1_7",
        name="baselines",
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

    res = run_baselines(cfg)
    payload = res["payload"]
    results = payload["feature_sets"]
    out_file = res["out_file"]

    print("\n=== Phase 1.7 Comparison Baselines (Table 10) Completed ===")
    print(f"Results written to: {out_file}")

    for fset in ["pso", "table2"]:
        data_res = results[fset]
        print(f"\n--- Feature Set: {fset.upper()} ({data_res['feature_count']} features) ---")
        for m_name, m_res in data_res["models"].items():
            met = m_res["metrics"]
            deltas = m_res["deltas_to_paper_table10"]
            auc_str = f"{met['roc_auc']:.4f}" if met.get("roc_auc") is not None else "N/A"
            delta_acc = deltas.get("accuracy", 0.0)
            print(
                f"  {m_name:16}: Acc={met['accuracy']:.4f} (Δ {delta_acc:+.4f}) | "
                f"F1={met['f1']:.4f} | Prec={met['precision']:.4f} | "
                f"Rec={met['recall']:.4f} | AUC={auc_str}"
            )
