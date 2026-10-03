"""Data cleaning, scaling, and preprocessing module (Pipeline A semantics).

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
  Section 4.1 (Data Collection Stage, deduplication context)
  Section 4.2.1 (Data Cleaning: missing data dropped)
  Section 4.2.2 (Standardization: z-score normalization)
  Figure 3 & Algorithm 6 (Pipeline execution order: clean -> standardize -> PSO -> SMOTE)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from src.data import (
    TokenFrequencyEncoder,
    load_raw,
    normalise_column_name,
    split_columns,
)


def clean_missing(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    """Clean missing values by dropping rows with NaN in numeric feature columns.

    Paper Reference: Section 4.2.1 (Data Cleaning) & Section 4.1.
    Numeric columns are dynamically identified as all feature columns in X except
    the categorical columns configured under data.categorical_columns.
    Rows with missing values only in categorical token columns are retained.

    Args:
        X: Feature matrix DataFrame.
        y: Binary label Series.
        meta: Identifier metadata DataFrame.
        cfg: Configuration dictionary.

    Returns:
        tuple of (X_clean, y_clean, meta_clean, report):
            - X_clean: Filtered feature matrix without numeric NaNs.
            - y_clean: Filtered binary labels.
            - meta_clean: Filtered metadata DataFrame.
            - report: Detailed cleaning statistics and comparison against paper references.
    """
    cat_cfg = cfg.get("data", {}).get("categorical_columns", [])
    norm_cat_cols = {normalise_column_name(c) for c in cat_cfg}
    numeric_cols = [
        c for c in X.columns if normalise_column_name(c) not in norm_cat_cols
    ]

    has_numeric_nan = X[numeric_cols].isna().any(axis=1)
    keep_mask = ~has_numeric_nan

    X_clean = X.loc[keep_mask].copy()
    y_clean = y.loc[keep_mask].copy()
    meta_clean = meta.loc[keep_mask].copy()

    rows_before = len(X)
    rows_after = len(X_clean)

    class_counts_before = {int(k): int(v) for k, v in y.value_counts().items()}
    class_counts_before.setdefault(0, 0)
    class_counts_before.setdefault(1, 0)

    class_counts_after = {int(k): int(v) for k, v in y_clean.value_counts().items()}
    class_counts_after.setdefault(0, 0)
    class_counts_after.setdefault(1, 0)

    dropped_counts_raw = y[has_numeric_nan].value_counts().to_dict()
    dropped_rows_by_class = {int(k): int(v) for k, v in dropped_counts_raw.items()}
    dropped_rows_by_class.setdefault(0, 0)
    dropped_rows_by_class.setdefault(1, 0)
    dropped_rows_total = int(has_numeric_nan.sum())

    numeric_columns_with_nan = [str(c) for c in numeric_cols if X[c].isna().any()]

    if len(numeric_columns_with_nan) > 0 and has_numeric_nan.any():
        dropped_numeric_slice = X.loc[has_numeric_nan, numeric_columns_with_nan]
        nan_counts_per_row = dropped_numeric_slice.isna().sum(axis=1)
        is_all_or_nothing = bool(
            (nan_counts_per_row == len(numeric_columns_with_nan)).all()
        )
    else:
        is_all_or_nothing = True

    paper_ref = cfg.get("paper_reference", {})
    paper_comparison = {}
    if "cleaned_rows" in paper_ref:
        paper_val = int(paper_ref["cleaned_rows"])
        paper_comparison["cleaned_rows"] = {
            "paper": paper_val,
            "actual": rows_after,
            "matches": bool(rows_after == paper_val),
            "delta": rows_after - paper_val,
        }
    if "cleaned_benign" in paper_ref:
        paper_val = int(paper_ref["cleaned_benign"])
        paper_comparison["cleaned_benign"] = {
            "paper": paper_val,
            "actual": class_counts_after[0],
            "matches": bool(class_counts_after[0] == paper_val),
            "delta": class_counts_after[0] - paper_val,
        }
    if "cleaned_suspicious" in paper_ref:
        paper_val = int(paper_ref["cleaned_suspicious"])
        paper_comparison["cleaned_suspicious"] = {
            "paper": paper_val,
            "actual": class_counts_after[1],
            "matches": bool(class_counts_after[1] == paper_val),
            "delta": class_counts_after[1] - paper_val,
        }
    if "dropped_rows" in paper_ref:
        paper_val = int(paper_ref["dropped_rows"])
        paper_comparison["dropped_rows"] = {
            "paper": paper_val,
            "actual": dropped_rows_total,
            "matches": bool(dropped_rows_total == paper_val),
            "delta": dropped_rows_total - paper_val,
        }

    report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "class_counts_before": class_counts_before,
        "class_counts_after": class_counts_after,
        "dropped_rows_by_class": dropped_rows_by_class,
        "dropped_rows_total": dropped_rows_total,
        "numeric_columns_with_nan": numeric_columns_with_nan,
        "is_all_or_nothing_missing": is_all_or_nothing,
        "paper_comparison": paper_comparison,
    }

    return X_clean, y_clean, meta_clean, report


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """Fit a z-score standard scaler on feature matrix X.

    Paper Reference: Section 4.2.2 (Standardization: z = (X - Mean) / std).
    Uses population standard deviation (ddof=0). Zero-variance features are
    assigned scale_ = 1.0 by StandardScaler and transformed to 0.0 without NaNs.

    Args:
        X: Numeric feature DataFrame.

    Returns:
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(X)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted scaler to feature matrix X preserving DataFrame structure.

    Args:
        scaler: Fitted StandardScaler instance.
        X: Numeric feature DataFrame to transform.

    Returns:
        Scaled DataFrame with identical index and column names.
    """
    scaled_arr = scaler.transform(X)
    return pd.DataFrame(scaled_arr, index=X.index, columns=X.columns)


if __name__ == "__main__":
    t0 = time.time()
    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Load raw data
    t_load_start = time.time()
    df, col_mapping = load_raw(cfg)
    t_load = time.time() - t_load_start

    # 2. Split columns
    t_split_start = time.time()
    X, y, meta = split_columns(df, cfg)
    t_split = time.time() - t_split_start

    # 3. Clean missing values
    t_clean_start = time.time()
    X_clean, y_clean, meta_clean, cleaning_report = clean_missing(X, y, meta, cfg)
    t_clean = time.time() - t_clean_start

    # 4. Token frequency encoding on cleaned data
    t_encode_start = time.time()
    encoder = TokenFrequencyEncoder(columns=cfg["data"]["categorical_columns"])
    X_clean_enc = encoder.fit_transform(X_clean)
    t_encode = time.time() - t_encode_start

    # 5. Fit and apply scaler (z-score)
    t_scale_start = time.time()
    scaler = fit_scaler(X_clean_enc)
    X_scaled = apply_scaler(scaler, X_clean_enc)
    t_scale = time.time() - t_scale_start

    t_total = time.time() - t0

    # Constant columns analysis
    variances = X_clean_enc.var(ddof=0)
    constant_cols = [str(c) for c in variances[variances == 0.0].index]
    non_constant_cols = [c for c in X_scaled.columns if c not in constant_cols]

    post_scaling_check = {
        "max_abs_mean_non_constant": float(
            X_scaled[non_constant_cols].mean().abs().max()
        ),
        "min_std_non_constant": float(
            X_scaled[non_constant_cols].std(ddof=0).min()
        ),
        "max_std_non_constant": float(
            X_scaled[non_constant_cols].std(ddof=0).max()
        ),
        "constant_columns_all_zero": bool(
            (X_scaled[constant_cols] == 0.0).all().all()
        ),
    }

    # Diagnostics: Policy comparison (computed, not applied)
    cat_cfg = cfg.get("data", {}).get("categorical_columns", [])
    norm_cat_cols = {normalise_column_name(c) for c in cat_cfg}
    numeric_cols = [
        c for c in X.columns if normalise_column_name(c) not in norm_cat_cols
    ]

    has_num_nan = X[numeric_cols].isna().any(axis=1)
    label_col = normalise_column_name(cfg["data"]["label_column"])

    p_any_nan = df.dropna()
    p_num_nan = df.loc[~has_num_nan]
    p_dup_v1 = p_num_nan.drop_duplicates(subset=X.columns, keep="first")
    p_dup_v2 = p_num_nan.drop_duplicates(subset=numeric_cols, keep="first")
    p_dup_addr = p_num_nan.drop_duplicates(subset=["address"], keep="first")

    policy_comparison = {
        "drop_any_nan": {
            "rows": len(p_any_nan),
            "benign": int((p_any_nan[label_col] == 0).sum()),
            "suspicious": int((p_any_nan[label_col] == 1).sum()),
        },
        "drop_numeric_nan": {
            "rows": len(p_num_nan),
            "benign": int((p_num_nan[label_col] == 0).sum()),
            "suspicious": int((p_num_nan[label_col] == 1).sum()),
        },
        "drop_numeric_nan_plus_dup_feature_v1": {
            "rows": len(p_dup_v1),
            "benign": int((p_dup_v1[label_col] == 0).sum()),
            "suspicious": int((p_dup_v1[label_col] == 1).sum()),
        },
        "drop_numeric_nan_plus_dup_feature_v2": {
            "rows": len(p_dup_v2),
            "benign": int((p_dup_v2[label_col] == 0).sum()),
            "suspicious": int((p_dup_v2[label_col] == 1).sum()),
        },
        "drop_numeric_nan_plus_dup_address": {
            "rows": len(p_dup_addr),
            "benign": int((p_dup_addr[label_col] == 0).sum()),
            "suspicious": int((p_dup_addr[label_col] == 1).sum()),
        },
    }

    duplicates_diag = {
        "raw_data": {
            "all_47_features": {
                int(k): int(v)
                for k, v in df[df.duplicated(subset=X.columns, keep="first")][
                    label_col
                ]
                .value_counts()
                .items()
            },
            "numeric_45_features": {
                int(k): int(v)
                for k, v in df[df.duplicated(subset=numeric_cols, keep="first")][
                    label_col
                ]
                .value_counts()
                .items()
            },
        },
        "cleaned_data": {
            "all_47_features": {
                int(k): int(v)
                for k, v in p_num_nan[
                    p_num_nan.duplicated(subset=X.columns, keep="first")
                ][label_col]
                .value_counts()
                .items()
            },
            "numeric_45_features": {
                int(k): int(v)
                for k, v in p_num_nan[
                    p_num_nan.duplicated(subset=numeric_cols, keep="first")
                ][label_col]
                .value_counts()
                .items()
            },
        },
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"phase1_2_cleaning_scaling_{timestamp}.json"

    result_payload = {
        "timestamp": timestamp,
        "config": cfg,
        "cleaning": cleaning_report,
        "constant_columns": constant_cols,
        "post_scaling_check": post_scaling_check,
        "timings_seconds": {
            "load_raw": t_load,
            "split_columns": t_split,
            "clean_missing": t_clean,
            "token_frequency_encoder": t_encode,
            "fit_apply_scaler": t_scale,
            "total": t_total,
        },
        "diagnostics": {
            "policy_comparison": policy_comparison,
            "duplicates": duplicates_diag,
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print(f"Cleaning and scaling completed. Results written to: {out_file}")
    print(f"Cleaned rows: {cleaning_report['rows_after']} (dropped: {cleaning_report['dropped_rows_total']})")
    print(f"Class counts after cleaning: {cleaning_report['class_counts_after']}")
    print(f"Constant columns count: {len(constant_cols)}")
    print("Paper comparison:")
    for k, v in cleaning_report["paper_comparison"].items():
        print(f"  {k}: paper={v['paper']}, actual={v['actual']}, match={v['matches']}, delta={v['delta']}")
