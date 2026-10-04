"""Master benchmark ladder aggregation module (Phase 2.0d).

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
- PLAN.md Phase 2.0 (Task 2.0d: Ladder benchmark table and diagnostics aggregation)

Reads existing results JSON files (Phase 1.8, Phase 2.0b, Phase 2.0c) and aggregates:
1. Leakage ladder master benchmark table across Paper, Pipeline A, Rung L1, and Rung B.
2. Per-rung test set diagnostics (size, balance, synthetic contamination, majority baseline, prevalence).
3. GA optimization summary table with inner-CV tuning estimate vs. test generalization.
4. Duplicate and train/test feature overlap diagnostic summary.

Outputs written:
- results/phase2_0d_ladder_benchmark_<timestamp>.json
- results/phase2_0d_ladder_benchmark_<timestamp>.csv
- results/phase2_0d_ladder_benchmark_<timestamp>.md
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_RESULT_PATTERNS = {
    "phase1_8_benchmark": "phase1_8_benchmark_*.json",
    "phase2_0b_ladder": "phase2_0b_pipeline_b_ladder_*.json",
    "phase2_0c_ga_l1_xgboost": "phase2_0c_ga_l1_xgboost_*.json",
    "phase2_0c_ga_b_xgboost": "phase2_0c_ga_b_xgboost_*.json",
    "phase2_0c_ga_b_svm": "phase2_0c_ga_b_svm_*.json",
    "phase2_0c_ga_b_isolation_forest": "phase2_0c_ga_b_isolation_forest_*.json",
    "phase2_0c_duplicates": "phase2_0c_duplicates_*.json",
}


def find_latest_file(results_dir: Path, pattern: str) -> Path:
    """Find the most recent file matching pattern in results_dir.

    Args:
        results_dir: Directory containing experiment result files.
        pattern: File glob pattern.

    Returns:
        Path to latest matching file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Missing required result file matching pattern '{pattern}' in {results_dir}. "
            "All prerequisite phases (1.8, 2.0b, 2.0c GA and duplicates) must be completed first."
        )
    return matches[-1]


def load_all_ladder_prerequisites(
    results_dir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load and parse all required result files for the ladder benchmark.

    Args:
        results_dir: Path to directory containing results.

    Returns:
        tuple of (data_dict, paths_dict).
    """
    data: dict[str, Any] = {}
    paths: dict[str, Path] = {}

    for key, pattern in REQUIRED_RESULT_PATTERNS.items():
        file_path = find_latest_file(results_dir, pattern)
        paths[key] = file_path
        with open(file_path, "r", encoding="utf-8") as f:
            data[key] = json.load(f)

    return data, paths


def build_per_rung_diagnostics(prereqs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract and compute dataset diagnostics across rungs (A, L1, B).

    Args:
        prereqs: Loaded prerequisite dictionary.

    Returns:
        dict mapping rung name ('pipeline_a', 'rung_l1', 'rung_b') to diagnostic stats.
    """
    p18_diag = prereqs["phase1_8_benchmark"].get("diagnostics", {})
    p20b_rungs = prereqs["phase2_0b_ladder"].get("rungs", {})

    l1_split = p20b_rungs.get("l1", {}).get("split_report", {})
    l1_maj = p20b_rungs.get("l1", {}).get("majority_baseline", {})

    b_split = p20b_rungs.get("b", {}).get("split_report", {})
    b_maj = p20b_rungs.get("b", {}).get("majority_baseline", {})

    # Pipeline A
    a_test_n = int(p18_diag.get("test_rows", 3065))
    a_test_benign = int(p18_diag.get("test_benign_rows", 1530))
    a_test_susp = int(p18_diag.get("test_suspicious_rows", 1535))
    a_syn_ratio = float(p18_diag.get("synthetic_test_ratio", 1245 / 3065))
    a_maj_acc = float(p18_diag.get("majority_class_accuracy", 1530 / 3065))
    a_prev = float(a_test_susp / a_test_n) if a_test_n > 0 else 0.0

    # Rung L1
    l1_test_n = int(l1_split.get("test_rows", 1803))
    l1_test_counts = l1_split.get("test_class_counts", {"0": 1533, "1": 270})
    l1_test_benign = int(l1_test_counts.get("0", l1_test_counts.get(0, 1533)))
    l1_test_susp = int(l1_test_counts.get("1", l1_test_counts.get(1, 270)))
    l1_syn_ratio = float(l1_split.get("synthetic_test_ratio", 0.0))
    l1_maj_acc = float(l1_maj.get("accuracy", 1533 / 1803))
    l1_maj_rec = float(l1_maj.get("recall_suspicious", 0.0))
    l1_prev = float(l1_test_susp / l1_test_n) if l1_test_n > 0 else 0.0

    # Rung B
    b_test_n = int(b_split.get("test_rows", 1803))
    b_test_counts = b_split.get("test_class_counts", {"0": 1533, "1": 270})
    b_test_benign = int(b_test_counts.get("0", b_test_counts.get(0, 1533)))
    b_test_susp = int(b_test_counts.get("1", b_test_counts.get(1, 270)))
    b_syn_ratio = float(b_split.get("synthetic_test_ratio", 0.0))
    b_maj_acc = float(b_maj.get("accuracy", 1533 / 1803))
    b_maj_rec = float(b_maj.get("recall_suspicious", 0.0))
    b_prev = float(b_test_susp / b_test_n) if b_test_n > 0 else 0.0

    return {
        "pipeline_a": {
            "name": "Pipeline A (Replication)",
            "test_size": a_test_n,
            "test_class_counts": {0: a_test_benign, 1: a_test_susp},
            "synthetic_fraction": a_syn_ratio,
            "majority_class_accuracy": a_maj_acc,
            "majority_class_suspicious_recall": 0.0,
            "prevalence": a_prev,
        },
        "rung_l1": {
            "name": "Rung L1 (SMOTE After Split Only)",
            "test_size": l1_test_n,
            "test_class_counts": {0: l1_test_benign, 1: l1_test_susp},
            "synthetic_fraction": l1_syn_ratio,
            "majority_class_accuracy": l1_maj_acc,
            "majority_class_suspicious_recall": l1_maj_rec,
            "prevalence": l1_prev,
        },
        "rung_b": {
            "name": "Rung B (Full Leakage-Safe)",
            "test_size": b_test_n,
            "test_class_counts": {0: b_test_benign, 1: b_test_susp},
            "synthetic_fraction": b_syn_ratio,
            "majority_class_accuracy": b_maj_acc,
            "majority_class_suspicious_recall": b_maj_rec,
            "prevalence": b_prev,
        },
    }


def build_master_table_rows(prereqs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build standardized master comparison rows for Model x Stage across rungs.

    Rows:
    1. XGBoost (Default)
    2. XGBoost (GA)
    3. SVM (Default)
    4. SVM (GA)
    5. Isolation Forest (Default)
    6. Isolation Forest (GA)
    7. CART (Default)
    8. CART (GA) -> not run
    9. LOF (Default)
    10. LOF (GA) -> not run

    Args:
        prereqs: Loaded prerequisite dictionary.

    Returns:
        List of row dictionaries containing metric comparisons across rungs.
    """
    p18_rows = {r["id"]: r for r in prereqs["phase1_8_benchmark"].get("rows", [])}
    p20b_rungs = prereqs["phase2_0b_ladder"].get("rungs", {})
    l1_models = p20b_rungs.get("l1", {}).get("models", {})
    b_models = p20b_rungs.get("b", {}).get("models", {})

    p20c_l1_xgb = prereqs["phase2_0c_ga_l1_xgboost"].get("test_metrics", {})
    p20c_b_xgb = prereqs["phase2_0c_ga_b_xgboost"].get("test_metrics", {})
    p20c_b_svm = prereqs["phase2_0c_ga_b_svm"].get("test_metrics", {})
    p20c_b_if = prereqs["phase2_0c_ga_b_isolation_forest"].get("test_metrics", {})

    row_definitions = [
        ("XGBoost", "Default", "xgboost_default", "xgboost", "default"),
        ("XGBoost", "GA Tuned", "xgboost_ga", "xgboost", "ga"),
        ("SVM", "Default", "svm_default", "svm", "default"),
        ("SVM", "GA Tuned", "svm_ga", "svm", "ga"),
        ("Isolation Forest", "Default", "isolation_forest_default", "isolation_forest", "default"),
        ("Isolation Forest", "GA Tuned", "isolation_forest_ga", "isolation_forest", "ga"),
        ("CART", "Default", "cart_baseline", "cart", "default"),
        ("CART", "GA Tuned", None, "cart", "ga"),
        ("LOF", "Default", "lof_baseline", "lof", "default"),
        ("LOF", "GA Tuned", None, "lof", "ga"),
    ]

    metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
    rows: list[dict[str, Any]] = []

    for model_name, stage_name, p18_id, model_key, stage_type in row_definitions:
        row: dict[str, Any] = {
            "model": model_name,
            "stage": stage_name,
        }

        # 1. Paper values
        paper_row = p18_rows.get(p18_id) if p18_id else None
        for m in metrics_list:
            if paper_row is None:
                row[f"paper_{m}"] = "not run"
            else:
                if m == "roc_auc":
                    val = paper_row.get("paper_auc")
                elif m in {"pr_auc", "mcc"}:
                    val = "not run"  # Paper did not compute or report PR-AUC / MCC
                else:
                    val = paper_row.get(f"paper_{m}")
                row[f"paper_{m}"] = val if val is not None else "not run"

        # 2. Pipeline A values (from Phase 1.8 benchmark)
        for m in metrics_list:
            if paper_row is None:
                row[f"pipeline_a_{m}"] = "not run"
            else:
                if m == "roc_auc":
                    val = paper_row.get("pipeline_a_pso_auc")
                elif m in {"pr_auc", "mcc"}:
                    val = "not run"  # Not tracked in master phase1_8 table
                else:
                    val = paper_row.get(f"pipeline_a_pso_{m}")
                row[f"pipeline_a_{m}"] = val if val is not None else "not run"

        # 3. Rung L1 values
        l1_met = None
        if stage_type == "default" and model_key in l1_models:
            l1_met = l1_models[model_key].get("metrics")
        elif stage_type == "ga" and model_key == "xgboost":
            l1_met = p20c_l1_xgb

        for m in metrics_list:
            if l1_met is not None and m in l1_met and l1_met[m] is not None:
                row[f"rung_l1_{m}"] = float(l1_met[m])
            else:
                row[f"rung_l1_{m}"] = "not run"

        # 4. Rung B values
        b_met = None
        if stage_type == "default" and model_key in b_models:
            b_met = b_models[model_key].get("metrics")
        elif stage_type == "ga":
            if model_key == "xgboost":
                b_met = p20c_b_xgb
            elif model_key == "svm":
                b_met = p20c_b_svm
            elif model_key == "isolation_forest":
                b_met = p20c_b_if

        for m in metrics_list:
            if b_met is not None and m in b_met and b_met[m] is not None:
                row[f"rung_b_{m}"] = float(b_met[m])
            else:
                row[f"rung_b_{m}"] = "not run"

        rows.append(row)

    return rows


def build_ga_summary_table(prereqs: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract GA optimization run parameters and compare tuning estimates vs. test results.

    Args:
        prereqs: Loaded prerequisite dictionary.

    Returns:
        List of GA run summary records.
    """
    ga_runs = [
        ("XGBoost", "Rung B", prereqs["phase2_0c_ga_b_xgboost"]),
        ("XGBoost", "Rung L1", prereqs["phase2_0c_ga_l1_xgboost"]),
        ("Isolation Forest", "Rung B", prereqs["phase2_0c_ga_b_isolation_forest"]),
        ("SVM", "Rung B", prereqs["phase2_0c_ga_b_svm"]),
    ]

    summary_rows: list[dict[str, Any]] = []

    for model, rung, data in ga_runs:
        evals = int(data.get("total_evaluations", 0))
        elapsed = float(data.get("elapsed_seconds", 0.0))
        metric_name = "f1_suspicious"
        best_fitness = float(data.get("best_fitness_cv_f1_suspicious", 0.0))
        best_params = data.get("best_params", {})
        test_f1 = float(data.get("test_metrics", {}).get("f1", 0.0))
        delta_cv_test = float(test_f1 - best_fitness)

        summary_rows.append({
            "model": model,
            "rung": rung,
            "evaluations": evals,
            "elapsed_seconds": elapsed,
            "ga_metric": metric_name,
            "best_inner_cv_fitness": best_fitness,
            "test_f1": test_f1,
            "tuning_vs_test_delta": delta_cv_test,
            "best_params": best_params,
        })

    return summary_rows


def build_duplicate_summary(prereqs: dict[str, Any]) -> dict[str, Any]:
    """Extract duplicate, contradiction, and train/test leakage diagnostic metrics.

    Args:
        prereqs: Loaded prerequisite dictionary.

    Returns:
        Dictionary of duplicate diagnostic figures.
    """
    p20c_dups = prereqs["phase2_0c_duplicates"]
    diag_summary = p20c_dups.get("diagnostic_summary", {})
    unseen_eval = p20c_dups.get("unseen_test_evaluation", {})
    full_eval = p20c_dups.get("full_test_evaluation", {})

    return {
        "duplicate_rows": diag_summary.get("duplicate_rows", {}),
        "contradictory_groups": diag_summary.get("contradictory_groups", {}),
        "train_test_overlap": diag_summary.get("train_test_overlap", {}),
        "full_test_metrics": full_eval.get("metrics", {}),
        "unseen_test_metrics": unseen_eval.get("metrics", {}),
        "unseen_majority_baseline": unseen_eval.get("majority_baseline", {}),
        "deltas_full_vs_unseen": p20c_dups.get("deltas_full_vs_unseen", {}),
    }


def format_val(val: Any, decimals: int = 4) -> str:
    """Format numeric or string value for Markdown/CSV tables."""
    if val is None or val == "not run":
        return "not run"
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def render_markdown_report(
    diagnostics: dict[str, dict[str, Any]],
    master_rows: list[dict[str, Any]],
    ga_summary: list[dict[str, Any]],
    dup_summary: dict[str, Any],
    input_paths: dict[str, Path],
    timestamp: str,
) -> str:
    """Render comprehensive Markdown benchmark report with footnotes and diagnostics."""
    lines: list[str] = [
        f"# Master Leakage Ladder Benchmark (Phase 2.0d)",
        f"",
        f"Generated at: {timestamp} (UTC).",
        f"Consolidates results across published literature, Pipeline A replication, and leakage-safe Pipeline B ladder rungs.",
        f"",
        f"## Source Files",
        f"",
    ]
    for key, path in sorted(input_paths.items()):
        lines.append(f"- **{key}**: `{path.name}`")

    # 1. Per-rung diagnostics
    lines.extend([
        f"",
        f"## 1. Per-Rung Evaluation Distribution Diagnostics",
        f"",
        f"| Evaluation Rung | Test Set Size | Benign (Class 0) | Suspicious (Class 1) | Synthetic Fraction | Majority Baseline Accuracy | Suspicious Recall | Prevalence |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r_key, r_info in diagnostics.items():
        counts = r_info["test_class_counts"]
        lines.append(
            f"| **{r_info['name']}** | {r_info['test_size']:,} | {counts[0]:,} | {counts[1]:,} | "
            f"{r_info['synthetic_fraction']*100:.2f}% | {r_info['majority_class_accuracy']:.4f} | "
            f"{r_info['majority_class_suspicious_recall']:.4f} | {r_info['prevalence']*100:.2f}% |"
        )

    # 2. Master Ladder Benchmark Table
    lines.extend([
        f"",
        f"## 2. Master Leakage Ladder Benchmark Table",
        f"",
        f"| Model | Stage | Metric | Paper | Pipeline A (Replication) | Rung L1 (SMOTE After Split) | Rung B (Full Leakage-Safe) |",
        f"| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    metrics_display = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall (Suspicious)", "recall"),
        ("F1-Score", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("MCC", "mcc"),
    ]

    for row in master_rows:
        m_name = row["model"]
        stg = row["stage"]
        first_metric = True
        for m_label, m_key in metrics_display:
            p_val = format_val(row[f"paper_{m_key}"])
            a_val = format_val(row[f"pipeline_a_{m_key}"])
            l1_val = format_val(row[f"rung_l1_{m_key}"])
            b_val = format_val(row[f"rung_b_{m_key}"])

            m_prefix = f"**{m_name}**" if first_metric else ""
            s_prefix = f"{stg}" if first_metric else ""
            lines.append(f"| {m_prefix:<18} | {s_prefix:<12} | {m_label:<20} | {p_val} | {a_val} | {l1_val} | {b_val} |")
            first_metric = False

    # Footnote
    lines.extend([
        f"",
        f"> [!NOTE]",
        f"> **Table Footnote**: Pipeline A's test set is balanced and contains synthetic rows (40.62% synthetic accounts), so accuracy and precision are not comparable across Pipeline A and Rungs L1/B. ROC-AUC and PR-AUC provide the closest methodological comparison, and PR-AUC is fundamentally dependent on test prevalence (50.08% in Pipeline A vs. 14.98% in L1/B).",
        f"",
    ])

    # 3. GA Optimization Summary Table
    lines.extend([
        f"## 3. Genetic Algorithm Optimization Summary",
        f"",
        f"| Model | Ladder Rung | Evaluations | Wall-Clock Time | GA Metric | Best Inner-CV Fitness | Test F1-Score | Tuning-Estimate vs. Test Delta | Best Hyperparameters |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for ga_r in ga_summary:
        params_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in ga_r["best_params"].items()
        )
        lines.append(
            f"| **{ga_r['model']}** | {ga_r['rung']} | {ga_r['evaluations']} | {ga_r['elapsed_seconds']:.2f}s | "
            f"`{ga_r['ga_metric']}` | {ga_r['best_inner_cv_fitness']:.4f} | {ga_r['test_f1']:.4f} | "
            f"{ga_r['tuning_vs_test_delta']:+.4f} | `{params_str}` |"
        )

    # 4. Duplicate & Feature Overlap Diagnostics
    dups = dup_summary["duplicate_rows"]
    contra = dup_summary["contradictory_groups"]
    overlap = dup_summary["train_test_overlap"]
    unseen_met = dup_summary["unseen_test_metrics"]
    full_met = dup_summary["full_test_metrics"]
    unseen_maj = dup_summary["unseen_majority_baseline"]
    deltas = dup_summary["deltas_full_vs_unseen"]

    def _c_cnt(d: dict[str, Any], cls_idx: int) -> int:
        return int(d.get(str(cls_idx), d.get(cls_idx, 0)))

    lines.extend([
        f"",
        f"## 4. Duplicate & Train/Test Overlap Diagnostic Summary",
        f"",
        f"- **Cleaned 47-Feature Matrix ($N=9,012$)**:",
        f"  - Redundant duplicate rows (`keep='first'`): **{dups.get('redundant_rows_count', 0)} ({dups.get('redundant_ratio', 0.0)*100:.2f}%)** [Benign: {_c_cnt(dups.get('redundant_by_class', {}), 0)}, Suspicious: {_c_cnt(dups.get('redundant_by_class', {}), 1)}]",
        f"  - Rows in duplicate clusters (`keep=False`): **{dups.get('cluster_rows_count', 0)} ({dups.get('cluster_ratio', 0.0)*100:.2f}%)** [Benign: {_c_cnt(dups.get('cluster_by_class', {}), 0)}, Suspicious: {_c_cnt(dups.get('cluster_by_class', {}), 1)}]",
        f"  - Contradictory groups (identical features with conflicting labels): **{contra.get('groups_count', 0)} groups ({contra.get('rows_count', 0)} rows)**",
        f"- **Train/Test Leakage Overlap (Test Split $N=1,803$)**:",
        f"  - Test rows appearing in train split: **{overlap.get('test_rows_in_train_count', 0)} ({overlap.get('test_rows_in_train_ratio', 0.0)*100:.2f}%)** [Benign: {_c_cnt(overlap.get('test_rows_in_train_by_class', {}), 0)}, Suspicious: {_c_cnt(overlap.get('test_rows_in_train_by_class', {}), 1)}]",
        f"  - Strictly unseen test rows: **{overlap.get('unseen_test_count', 0)} ({overlap.get('unseen_test_ratio', 0.0)*100:.2f}%)** [Benign: {_c_cnt(overlap.get('unseen_test_by_class', {}), 0)}, Suspicious: {_c_cnt(overlap.get('unseen_test_by_class', {}), 1)}]",
        f"  - Unseen test set majority baseline accuracy: **{unseen_maj.get('accuracy', 0.0):.4f}** ({unseen_maj.get('accuracy', 0.0)*100:.2f}%)",
        f"- **XGBoost Performance on Unseen Test Rows vs. Full Test Set**:",
        f"  - Accuracy: {unseen_met.get('accuracy', 0.0):.4f} (Full: {full_met.get('accuracy', 0.0):.4f}, Delta: {deltas.get('accuracy', 0.0):+.4f})",
        f"  - Precision: {unseen_met.get('precision', 0.0):.4f} (Full: {full_met.get('precision', 0.0):.4f}, Delta: {deltas.get('precision', 0.0):+.4f})",
        f"  - Recall (Suspicious): {unseen_met.get('recall', 0.0):.4f} (Full: {full_met.get('recall', 0.0):.4f}, Delta: {deltas.get('recall', 0.0):+.4f})",
        f"  - F1-Score: {unseen_met.get('f1', 0.0):.4f} (Full: {full_met.get('f1', 0.0):.4f}, Delta: {deltas.get('f1', 0.0):+.4f})",
        f"  - ROC-AUC: {unseen_met.get('roc_auc', 0.0):.4f} (Full: {full_met.get('roc_auc', 0.0):.4f}, Delta: {deltas.get('roc_auc', 0.0):+.4f})",
        f"  - PR-AUC: {unseen_met.get('pr_auc', 0.0):.4f} (Full: {full_met.get('pr_auc', 0.0):.4f}, Delta: {deltas.get('pr_auc', 0.0):+.4f})",
        f"  - MCC: {unseen_met.get('mcc', 0.0):.4f} (Full: {full_met.get('mcc', 0.0):.4f}, Delta: {deltas.get('mcc', 0.0):+.4f})",
    ])

    return "\n".join(lines)


def run_benchmark_ladder(results_dir: Path | str = "results") -> dict[str, Any]:
    """Execute master benchmark ladder aggregation across all prerequisite runs.

    Args:
        results_dir: Path to directory containing results.

    Returns:
        Dictionary of generated artifact paths and report payloads.
    """
    res_path = Path(results_dir)
    prereqs, input_paths = load_all_ladder_prerequisites(res_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    diagnostics = build_per_rung_diagnostics(prereqs)
    master_rows = build_master_table_rows(prereqs)
    ga_summary = build_ga_summary_table(prereqs)
    dup_summary = build_duplicate_summary(prereqs)

    payload = {
        "timestamp": timestamp,
        "input_files": {k: str(p) for k, p in input_paths.items()},
        "diagnostics": diagnostics,
        "master_table": master_rows,
        "ga_summary": ga_summary,
        "duplicate_diagnostics": dup_summary,
    }

    # 1. Write JSON
    out_json = res_path / f"phase2_0d_ladder_benchmark_{timestamp}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 2. Write CSV
    out_csv = res_path / f"phase2_0d_ladder_benchmark_{timestamp}.csv"
    if master_rows:
        fieldnames = list(master_rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in master_rows:
                writer.writerow(r)

    # 3. Write Markdown
    out_md = res_path / f"phase2_0d_ladder_benchmark_{timestamp}.md"
    md_content = render_markdown_report(
        diagnostics, master_rows, ga_summary, dup_summary, input_paths, timestamp
    )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "out_json": out_json,
        "out_csv": out_csv,
        "out_md": out_md,
        "payload": payload,
        "markdown": md_content,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate master benchmark ladder table and diagnostics (Phase 2.0d)."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing prerequisite result files (default: results).",
    )
    args = parser.parse_args()

    res = run_benchmark_ladder(args.results_dir)
    print(res["markdown"])
    print(f"\nArtifacts generated:")
    print(f"  JSON: {res['out_json']}")
    print(f"  CSV:  {res['out_csv']}")
    print(f"  MD:   {res['out_md']}")
