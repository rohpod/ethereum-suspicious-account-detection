"""Data loading, column splitting, token encoding, and dataset validation.

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
  Section 4.1 (Data Collection Stage) & Appendix A Table A2 (Dataset Features).
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin


def normalise_column_name(col: str) -> str:
    """Normalise a column name.

    Strips whitespace, converts to lowercase, replaces non-alphanumeric sequences
    with single underscores, and strips leading/trailing underscores.

    Args:
        col: Original column name.

    Returns:
        Normalised column name.
    """
    cleaned = col.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def load_raw(
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load raw Ethereum transactions CSV dataset and normalise column names.

    Paper Reference: Section 4.1 (Data Collection Stage).

    Args:
        cfg: Configuration dictionary containing paths and data parameters.
             If None, loads from config/config.yaml.

    Returns:
        tuple of:
            - df: DataFrame with normalised column names.
            - mapping: Dictionary mapping original column names to normalised names.
    """
    if cfg is None:
        cfg_path = Path("config/config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_raw"])
    filename = cfg["data"]["filename"]
    csv_path = data_dir / filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

    df_raw = pd.read_csv(csv_path)

    # Build mapping and verify uniqueness
    mapping = {col: normalise_column_name(col) for col in df_raw.columns}
    normalised_names = list(mapping.values())
    if len(normalised_names) != len(set(normalised_names)):
        duplicates = [c for c in normalised_names if normalised_names.count(c) > 1]
        raise ValueError(f"Duplicate normalised column names found: {set(duplicates)}")

    df = df_raw.rename(columns=mapping)
    return df, mapping


def split_columns(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split DataFrame into feature matrix X, label series y, and identifier metadata.

    Paper Reference: Section 4.1 & Table A2. Index, Address, and Flag columns
    must never enter the feature matrix X.

    Args:
        df: DataFrame with normalised column names.
        cfg: Configuration dictionary.

    Returns:
        tuple of (X, y, meta):
            - X: DataFrame containing only features.
            - y: Series containing binary labels (0/1).
            - meta: DataFrame containing identifier columns (Unnamed: 0, Index, Address).

    Raises:
        ValueError: If label column is missing, not binary, or if any forbidden
                    identifier/label column is present in X.
    """
    # Identify label column
    raw_label_col = cfg["data"]["label_column"]
    norm_label_col = normalise_column_name(raw_label_col)

    if norm_label_col in df.columns:
        label_col = norm_label_col
    elif raw_label_col in df.columns:
        label_col = raw_label_col
    else:
        raise ValueError(
            f"Label column '{raw_label_col}' (normalised '{norm_label_col}') not found in DataFrame."
        )

    y = df[label_col].copy()
    unique_labels = set(y.dropna().unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Label column '{label_col}' must contain binary values (0/1), got: {unique_labels}"
        )
    y = y.astype(int)

    # Identify ID/metadata columns
    raw_id_cols = cfg["data"]["id_columns"]
    norm_id_cols = [normalise_column_name(c) for c in raw_id_cols]

    matched_id_cols = []
    for raw_c, norm_c in zip(raw_id_cols, norm_id_cols):
        if norm_c in df.columns:
            matched_id_cols.append(norm_c)
        elif raw_c in df.columns:
            matched_id_cols.append(raw_c)

    meta = df[matched_id_cols].copy()

    # Drop id columns and label column to produce X
    drop_cols = list(set(matched_id_cols + [label_col]))
    X = df.drop(columns=drop_cols).copy()

    # Strict check: Index, Address, Flag, Unnamed: 0 must never be in X
    forbidden_exact = {"index", "address", "flag", "unnamed_0"}
    for col in X.columns:
        norm_c = normalise_column_name(col)
        if norm_c in forbidden_exact or norm_c.startswith("unnamed"):
            raise ValueError(
                f"Forbidden identifier/label column found in feature matrix X: '{col}'"
            )

    return X, y, meta


class TokenFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoder for ERC20 token type columns (sklearn-style).

    Paper Reference: Section 4.1 & Table A2 ('Most Sent Token Type (ERC20)',
    'Most Received Token Type (ERC20)').

    Transformations:
    - String whitespace stripped.
    - Missing (NaN/None) mapped to '__missing__'.
    - Empty or whitespace-only mapped to '__blank__'.
    - '0' preserved as its own category.
    - Category frequency = category count / n_fit_rows.
    - Unseen categories during transform mapped to 0.0.
    - Replaces values in-place (column names unchanged).
    - Selects columns strictly by NAME.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self.frequencies_: dict[str, dict[str, float]] = {}
        self.resolved_columns_: list[str] = []

    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is None:
            candidates = [
                "erc20_most_sent_token_type",
                "erc20_most_rec_token_type",
                " ERC20 most sent token type",
                " ERC20_most_rec_token_type",
            ]
            return [c for c in candidates if c in X.columns]

        resolved = []
        for col in self.columns:
            if col in X.columns:
                resolved.append(col)
            else:
                norm_c = normalise_column_name(col)
                if norm_c in X.columns:
                    resolved.append(norm_c)
                else:
                    raise KeyError(
                        f"Column '{col}' (normalised '{norm_c}') not found in DataFrame."
                    )
        return resolved

    @staticmethod
    def _clean_series(s: pd.Series) -> pd.Series:
        """Standardise category values for token columns."""
        def _clean_val(val: Any) -> str:
            if pd.isna(val):
                return "__missing__"
            val_str = str(val).strip()
            if val_str == "":
                return "__blank__"
            if val_str == "0":
                return "0"
            return val_str

        return s.map(_clean_val).astype("string")

    def fit(self, X: pd.DataFrame, y: Any = None) -> TokenFrequencyEncoder:
        """Compute frequency mappings on training data."""
        self.resolved_columns_ = self._resolve_columns(X)
        self.frequencies_ = {}
        n_rows = len(X)

        if n_rows == 0:
            raise ValueError("Cannot fit TokenFrequencyEncoder on empty DataFrame.")

        for col in self.resolved_columns_:
            cleaned = self._clean_series(X[col])
            counts = cleaned.value_counts()
            freq_dict = (counts / n_rows).to_dict()
            self.frequencies_[col] = {str(k): float(v) for k, v in freq_dict.items()}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace token column values with frequencies (column names unchanged)."""
        if not self.frequencies_:
            raise ValueError("TokenFrequencyEncoder has not been fitted yet.")

        X_out = X.copy()
        for col in self.resolved_columns_:
            cleaned = self._clean_series(X_out[col])
            freq_map = self.frequencies_[col]
            # Unseen categories get 0.0
            X_out[col] = cleaned.map(
                lambda c, fmap=freq_map: fmap.get(str(c), 0.0)
            ).astype(float)

        return X_out


def validate(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Validate dataset properties and compare against reference values from paper.

    Paper Reference: Section 4.1 & Table A2.
    Does NOT raise on paper count mismatches (reports them).
    Only raises if label column is missing or non-binary.

    Returns:
        dict containing validation metrics and paper comparison results.
    """
    unique_labels = set(y.dropna().unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Label column must be binary (0/1), got: {unique_labels}")

    rows = len(df)
    class_counts_raw = y.value_counts().to_dict()
    class_counts = {int(k): int(v) for k, v in class_counts_raw.items()}
    class_counts.setdefault(0, 0)
    class_counts.setdefault(1, 0)

    feature_count = int(X.shape[1])
    total_columns = int(df.shape[1])

    # Columns minus Address and FLAG
    addr_flag_cols = [
        c
        for c in df.columns
        if normalise_column_name(c) in {"address", "flag"}
        or c in {cfg["data"]["label_column"], "Address"}
    ]
    columns_minus_address_flag = int(df.shape[1] - len(set(addr_flag_cols)))

    # Missing values
    missing_cells = int(df.isna().sum().sum())
    missing_rows = int(df.isna().any(axis=1).sum())
    missing_columns = int(df.isna().any(axis=0).sum())
    per_column_missing = {
        str(col): int(count)
        for col, count in df.isna().sum().items()
        if count > 0
    }

    # Rows with any NaN by class
    any_nan_mask = df.isna().any(axis=1)
    any_nan_class = y[any_nan_mask].value_counts().to_dict()
    rows_with_any_nan_by_class = {
        0: int(any_nan_class.get(0, 0)),
        1: int(any_nan_class.get(1, 0)),
    }

    # Rows with NaN in numeric columns only by class
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    num_nan_mask = df[numeric_cols].isna().any(axis=1)
    num_nan_class = y[num_nan_mask].value_counts().to_dict()
    rows_with_nan_in_numeric_columns_only_by_class = {
        0: int(num_nan_class.get(0, 0)),
        1: int(num_nan_class.get(1, 0)),
    }

    # Whitespace-only counts in token columns
    cat_cfg = cfg.get("data", {}).get("categorical_columns", [])
    token_cols = []
    for c in cat_cfg:
        norm_c = normalise_column_name(c)
        if norm_c in df.columns:
            token_cols.append(norm_c)
        elif c in df.columns:
            token_cols.append(c)

    whitespace_only_counts = {}
    for tc in token_cols:
        non_null_s = df[tc].dropna().astype(str)
        whitespace_count = int((non_null_s.str.strip() == "").sum())
        whitespace_only_counts[str(tc)] = whitespace_count

    # Constant columns (<= 1 unique non-null value)
    constant_columns = [
        str(col) for col in X.columns if X[col].dropna().nunique() <= 1
    ]

    # Duplicate addresses
    addr_candidates = [c for c in meta.columns if "address" in c.lower()]
    if addr_candidates:
        addr_col = addr_candidates[0]
        dup_addr_count = int(meta[addr_col].duplicated().sum())
        addr_has_both_labels = bool(
            (df.groupby(meta[addr_col])[y.name].nunique() > 1).any()
        )
    else:
        dup_addr_count = 0
        addr_has_both_labels = False

    # Duplicate rows over feature columns + label EXCLUDING id columns
    feat_and_label = pd.concat([X, y], axis=1)
    dup_rows_feat_label = int(feat_and_label.duplicated().sum())

    # Paper reference comparisons
    paper_ref = cfg.get("paper_reference", {})
    paper_comparison = {}
    if "rows" in paper_ref:
        paper_rows = int(paper_ref["rows"])
        paper_comparison["rows"] = {
            "paper": paper_rows,
            "actual": rows,
            "matches": bool(rows == paper_rows),
        }
    if "benign" in paper_ref:
        paper_benign = int(paper_ref["benign"])
        paper_comparison["benign"] = {
            "paper": paper_benign,
            "actual": class_counts[0],
            "matches": bool(class_counts[0] == paper_benign),
        }
    if "suspicious" in paper_ref:
        paper_suspicious = int(paper_ref["suspicious"])
        paper_comparison["suspicious"] = {
            "paper": paper_suspicious,
            "actual": class_counts[1],
            "matches": bool(class_counts[1] == paper_suspicious),
        }
    if "features_text" in paper_ref:
        paper_feat_txt = int(paper_ref["features_text"])
        paper_comparison["features_text"] = {
            "paper": paper_feat_txt,
            "actual": feature_count,
            "matches": bool(feature_count == paper_feat_txt),
        }
    if "features_table_a2" in paper_ref:
        paper_feat_a2 = int(paper_ref["features_table_a2"])
        paper_comparison["features_table_a2"] = {
            "paper": paper_feat_a2,
            "actual": feature_count,
            "matches": bool(feature_count == paper_feat_a2),
        }

    return {
        "rows": rows,
        "class_counts": class_counts,
        "feature_count": feature_count,
        "total_columns": total_columns,
        "columns_minus_address_flag": columns_minus_address_flag,
        "missing_cells": missing_cells,
        "missing_rows": missing_rows,
        "missing_columns": missing_columns,
        "per_column_missing": per_column_missing,
        "rows_with_any_nan_by_class": rows_with_any_nan_by_class,
        "rows_with_nan_in_numeric_columns_only_by_class": rows_with_nan_in_numeric_columns_only_by_class,
        "whitespace_only_counts_in_token_columns": whitespace_only_counts,
        "constant_columns": constant_columns,
        "duplicate_address_count": dup_addr_count,
        "address_has_both_labels": addr_has_both_labels,
        "duplicate_rows_over_features_and_label": dup_rows_feat_label,
        "paper_comparison": paper_comparison,
    }


if __name__ == "__main__":
    t0 = time.time()
    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Load
    t_load_start = time.time()
    df, col_mapping = load_raw(cfg)
    t_load = time.time() - t_load_start

    # 2. Split
    t_split_start = time.time()
    X, y, meta = split_columns(df, cfg)
    t_split = time.time() - t_split_start

    # 3. Validate
    t_val_start = time.time()
    val_report = validate(df, X, y, meta, cfg)
    t_val = time.time() - t_val_start

    # 4. Encode (Pipeline A semantics: fit/transform on all data)
    t_enc_start = time.time()
    encoder = TokenFrequencyEncoder(columns=cfg["data"]["categorical_columns"])
    X_encoded = encoder.fit_transform(X)
    t_enc = time.time() - t_enc_start

    t_total = time.time() - t0

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"phase1_1_data_validation_{timestamp}.json"

    result_payload = {
        "timestamp": timestamp,
        "config": cfg,
        "validation": val_report,
        "column_mapping": col_mapping,
        "timings_seconds": {
            "load_raw": t_load,
            "split_columns": t_split,
            "validate": t_val,
            "token_frequency_encoder": t_enc,
            "total": t_total,
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print(f"Validation completed successfully. Results written to: {out_file}")
    print(f"Rows: {val_report['rows']}, Features: {val_report['feature_count']}")
    print(f"Class counts: {val_report['class_counts']}")
    print("Paper comparison:")
    for k, v in val_report["paper_comparison"].items():
        print(f"  {k}: paper={v['paper']}, actual={v['actual']}, match={v['matches']}")
