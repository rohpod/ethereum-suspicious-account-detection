"""Model builders and inference wrappers for XGBoost, SVM, and Isolation Forest.

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
  Section 4.3 (Classification Stage)
  Table 1 (Hyperparameters overview)
  Tables 5-7 (Hyperparameter defaults: XGBoost, SVM, Isolation Forest)
  Algorithms 3-5 (Model algorithms)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC


def _load_default_cfg() -> dict[str, Any]:
    """Helper to load config/config.yaml if no config is supplied."""
    cfg_path = Path("config/config.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def build_xgboost(
    params: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> xgb.XGBClassifier:
    """Build an unfitted XGBoost classifier with paper defaults.

    Paper Reference: Table 1, Table 5, Algorithm 3.
    Defaults from Table 5: n_estimators=100, gamma=0.0, min_child_weight=1,
    colsample_bytree=1.0, max_depth=6, reg_lambda=1.0, learning_rate=0.3.
    Objective: binary:logistic, subsample=1.0.

    Args:
        params: Optional hyperparameter overrides.
        cfg: Optional configuration dictionary.

    Returns:
        Unfitted XGBClassifier instance.
    """
    if cfg is None:
        cfg = _load_default_cfg()

    model_cfg = cfg.get("models", {}).get("xgboost", {})
    default_params = model_cfg.get("defaults", {}).copy()

    # Base parameters
    init_params: dict[str, Any] = {
        "n_estimators": default_params.get("n_estimators", 100),
        "gamma": default_params.get("gamma", 0.0),
        "min_child_weight": default_params.get("min_child_weight", 1),
        "colsample_bytree": default_params.get("colsample_bytree", 1.0),
        "max_depth": default_params.get("max_depth", 6),
        "reg_lambda": default_params.get("reg_lambda", 1.0),
        "learning_rate": default_params.get("learning_rate", 0.3),
        "objective": model_cfg.get("objective", "binary:logistic"),
        "subsample": model_cfg.get("subsample", 1.0),
        "random_state": cfg.get("seed", 42),
    }

    if params:
        init_params.update(params)

    return xgb.XGBClassifier(**init_params)


def build_svm(
    params: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> SVC:
    """Build an unfitted Support Vector Machine (SVC) classifier with paper defaults.

    Paper Reference: Table 1, Table 6, Algorithm 4.
    Defaults from Table 6: C=0.1, gamma=0.1. Kernel: rbf. probability=False.

    Args:
        params: Optional hyperparameter overrides.
        cfg: Optional configuration dictionary.

    Returns:
        Unfitted SVC instance.
    """
    if cfg is None:
        cfg = _load_default_cfg()

    model_cfg = cfg.get("models", {}).get("svm", {})
    default_params = model_cfg.get("defaults", {}).copy()

    init_params: dict[str, Any] = {
        "C": default_params.get("C", 0.1),
        "gamma": default_params.get("gamma", 0.1),
        "kernel": model_cfg.get("kernel", "rbf"),
        "probability": False,
        "random_state": cfg.get("seed", 42),
    }

    if params:
        init_params.update(params)

    return SVC(**init_params)


def build_isolation_forest(
    params: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> IsolationForest:
    """Build an unfitted Isolation Forest with paper defaults.

    Paper Reference: Table 1, Table 7, Algorithm 5.
    Defaults from Table 7: contamination=0.1, max_samples=256, n_estimators=100.

    Args:
        params: Optional hyperparameter overrides.
        cfg: Optional configuration dictionary.

    Returns:
        Unfitted IsolationForest instance.
    """
    if cfg is None:
        cfg = _load_default_cfg()

    model_cfg = cfg.get("models", {}).get("isolation_forest", {})
    default_params = model_cfg.get("defaults", {}).copy()

    init_params: dict[str, Any] = {
        "contamination": default_params.get("contamination", 0.1),
        "max_samples": default_params.get("max_samples", 256),
        "n_estimators": default_params.get("n_estimators", 100),
        "random_state": cfg.get("seed", 42),
    }

    if params:
        init_params.update(params)

    return IsolationForest(**init_params)


def fit_predict_scores(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit model on training data and return binary predictions and continuous anomaly scores.

    Predictions y_pred are binary {0, 1} where 1 represents suspicious.
    Scores are aligned such that higher values indicate higher suspiciousness:
    - XGBoost: model.predict_proba(X_test)[:, 1]
    - SVM: model.decision_function(X_test)
    - Isolation Forest: fitted unsupervised on X_train only (labels ignored),
      predict -1 -> 1 (suspicious) and +1 -> 0 (benign),
      scores = -model.score_samples(X_test).

    Args:
        name: Model identifier ('xgboost', 'svm', or 'isolation_forest').
        model: Estimator instance to fit.
        X_train: Training feature DataFrame.
        y_train: Training binary label Series.
        X_test: Test feature DataFrame.

    Returns:
        tuple of (y_pred, scores):
            - y_pred: 1D np.ndarray of binary {0, 1} predictions.
            - scores: 1D np.ndarray of continuous scores (higher = more suspicious).
    """
    norm_name = name.strip().lower().replace("_", "").replace("-", "")

    if norm_name in {"xgboost", "xgb"}:
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=int)
        scores = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
        return y_pred, scores

    elif norm_name in {"svm", "svc"}:
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=int)
        scores = np.asarray(model.decision_function(X_test), dtype=float)
        return y_pred, scores

    elif norm_name in {"isolationforest", "if", "isoforest"}:
        # Unsupervised fitting: labels are ignored entirely
        model.fit(X_train)
        raw_pred = model.predict(X_test)
        # raw_pred: -1 = anomaly (suspicious), +1 = inlier (benign)
        y_pred = np.where(raw_pred == -1, 1, 0).astype(int)
        # score_samples: more negative = more abnormal. Negating gives higher = more suspicious.
        scores = -np.asarray(model.score_samples(X_test), dtype=float)
        return y_pred, scores

    else:
        raise ValueError(
            f"Unsupported model name '{name}'. Expected one of: 'xgboost', 'svm', 'isolation_forest'."
        )
