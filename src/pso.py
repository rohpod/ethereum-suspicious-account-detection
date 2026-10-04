"""Particle Swarm Optimization (PSO) feature selection module (Pipeline A semantics).

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
  Section 4.2.3 (Feature Selection Using Particle Swarm Optimization)
  Equations 2-11 (Particle representation, velocity limits, inertia weight, updates)
  Algorithm 1 (PSO algorithm) & Figure 4 (Flowchart)
  Table 2 (Selected features after PSO)
  Algorithm 6 (Full detection framework, Pipeline A)

Algorithm 1 Documented Interpretations:
1. Fitness Optimization Direction: The paper text in Section 4.2.3 states "search
   for the minimum objective function value x** for each i = 1, 2, ..., n_p", and
   reports achieving a Root Mean Square Error (RMSE) of 0.3443 after 50 iterations.
   Therefore, fitness is MINIMISED as RMSE. The '>' sign in Algorithm 1 line 5 is
   read as a typographical error in the paper's pseudocode.
2. Global Best Tracking: Algorithm 1 line 7 ('x**_i = x*_i') inside the particle loop
   is read as the standard swarm-wide global best g*, updating g* whenever any
   particle's personal best improves upon it.
3. Dimension Loop: Algorithm 1 line 9 ('For d = 1 to iteration') is read as a loop
   over feature dimensions d = 1 to n_features (D), not a redundant re-iteration loop.

Pipeline A Semantics:
In Pipeline A, PSO feature selection is executed on the entire cleaned and scaled
dataset (all 9,012 accounts) before SMOTE and data splitting (Algorithm 6 Step 2).
Data is passed as arguments into run_pso(X, y, cfg) with no data loading inside the
function, allowing re-use on training folds in Phase 2.0 (Pipeline B).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from src.data import (
    TokenFrequencyEncoder,
    load_raw,
    normalise_column_name,
    split_columns,
)
from src.evaluate import write_results
from src.preprocess import apply_scaler, clean_missing, fit_scaler


def hash_index(idx: Any) -> str:
    """Compute deterministic SHA-256 hash of sorted index values.

    Args:
        idx: Iterable index (e.g., pd.Index, np.ndarray, list).

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    sorted_vals = sorted(list(idx))
    payload = json.dumps([str(v) for v in sorted_vals]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_fold_smote(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    k_neighbors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to training fold with small-minority guard.

    Guard: if minority count <= k_neighbors, reduce k to minority_count - 1;
    if minority_count < 2, skip SMOTE for that fold. Log nothing noisy.
    """
    y_arr = np.asarray(y_tr, dtype=int)
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        return X_tr, y_tr

    minority_count = int(np.min(counts))
    if minority_count < 2:
        return X_tr, y_tr

    k = k_neighbors
    if minority_count <= k:
        k = minority_count - 1

    smote = SMOTE(k_neighbors=k, random_state=seed)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)
    return np.asarray(X_res, dtype=float), np.asarray(y_res, dtype=int)


def pso_update_particle(
    x: np.ndarray,
    v: np.ndarray,
    pbest: np.ndarray,
    gbest: np.ndarray,
    w: float,
    c1: float,
    c2: float,
    v_max: float,
    bounds: tuple[float, float],
    r1: np.ndarray,
    r2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Update a single particle's velocity and position according to Eq. 10 & 11.

    Clamps velocity to [-v_max, v_max] per Eq. 6-7, 14-15.
    Clips position to bounds [bounds[0], bounds[1]] per Eq. 18-19.

    Args:
        x: Current particle position array of shape (D,).
        v: Current particle velocity array of shape (D,).
        pbest: Personal best position array of shape (D,).
        gbest: Swarm global best position array of shape (D,).
        w: Inertia weight at current iteration.
        c1: Cognitive acceleration coefficient.
        c2: Social acceleration coefficient.
        v_max: Maximum allowable velocity magnitude.
        bounds: (min_bound, max_bound) for position coordinates.
        r1: Uniform random array in [0, 1] of shape (D,).
        r2: Uniform random array in [0, 1] of shape (D,).

    Returns:
        tuple of (new_position, new_velocity).
    """
    new_v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
    new_v = np.clip(new_v, -v_max, v_max)
    new_x = x + new_v
    new_x = np.clip(new_x, bounds[0], bounds[1])
    return new_x, new_v


class RMSEFitness:
    """Stratified K-fold cross-validated RMSE fitness evaluator using DecisionTreeClassifier."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> None:
        self.eval_count = 0
        pso_cfg = cfg.get("pso", {})
        fitness_cfg = pso_cfg.get("fitness", {})

        self.empty_subset_fitness = float(
            pso_cfg.get("empty_subset_fitness", 1.0)
        )
        self.cv_folds = int(fitness_cfg.get("cv_folds", 3))
        self.smote_in_fold = bool(fitness_cfg.get("smote_in_fold", False))
        self.seed = int(cfg.get("seed", 42))
        smote_cfg = cfg.get("smote", {})
        self.k_neighbors = int(smote_cfg.get("k_neighbors", 5))

        self.X_arr = np.asarray(X, dtype=float)
        self.y_arr = np.asarray(y, dtype=int)

        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.seed,
        )
        self.splits = list(skf.split(self.X_arr, self.y_arr))

    def __call__(self, mask: np.ndarray | list[bool]) -> float:
        """Evaluate stratified cross-validation RMSE of selected features.

        Args:
            mask: Boolean feature selection mask of length n_features.

        Returns:
            Mean RMSE across CV folds, or empty_subset_fitness if no features selected.
        """
        self.eval_count += 1
        mask_arr = np.asarray(mask, dtype=bool)
        if not np.any(mask_arr):
            return self.empty_subset_fitness

        col_indices = np.where(mask_arr)[0]
        fold_rmses: list[float] = []

        for train_idx, val_idx in self.splits:
            X_fold_tr = self.X_arr[train_idx][:, col_indices]
            y_fold_tr = self.y_arr[train_idx]

            if self.smote_in_fold:
                X_fold_tr, y_fold_tr = apply_fold_smote(
                    X_fold_tr, y_fold_tr, self.k_neighbors, self.seed
                )

            clf = DecisionTreeClassifier(random_state=self.seed)
            clf.fit(X_fold_tr, y_fold_tr)
            preds = clf.predict(self.X_arr[val_idx][:, col_indices])
            rmse = float(np.sqrt(np.mean((self.y_arr[val_idx] - preds) ** 2)))
            fold_rmses.append(rmse)

        return float(np.mean(fold_rmses))


def make_rmse_fitness(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
) -> RMSEFitness:
    """Create a Stratified K-fold cross-validated RMSE fitness evaluator.

    Args:
        X: Feature matrix DataFrame.
        y: Binary target Series.
        cfg: Configuration dictionary.

    Returns:
        Callable RMSEFitness instance with eval_count tracking.
    """
    return RMSEFitness(X, y, cfg)


def compute_pso_cache_hash(
    cfg: dict[str, Any],
    X_shape: tuple[int, int],
    columns: list[str],
    seed: int,
    pipeline: str = "pipeline_a",
    train_index_hash: str | None = None,
) -> str:
    """Compute a deterministic SHA-256 hash for caching PSO results.

    Args:
        cfg: Configuration dictionary containing PSO parameters.
        X_shape: Tuple of (n_rows, n_cols).
        columns: List of feature column names.
        seed: Random seed.
        pipeline: Pipeline identifier (default "pipeline_a").
        train_index_hash: Optional SHA-256 hash of training index.

    Returns:
        Hexadecimal hash string.
    """
    pso_cfg = copy.deepcopy(cfg.get("pso", {}))
    if pipeline == "pipeline_a":
        if "fitness" in pso_cfg and "smote_in_fold" in pso_cfg["fitness"]:
            del pso_cfg["fitness"]["smote_in_fold"]

    payload: dict[str, Any] = {
        "pso_config": pso_cfg,
        "data_shape": list(X_shape),
        "column_names": list(columns),
        "seed": seed,
    }
    if pipeline != "pipeline_a" or train_index_hash is not None:
        payload["pipeline"] = pipeline
        payload["train_index_hash"] = train_index_hash

    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_with_table2(
    selected: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compare selected features with Table 2 from El-Attar et al.

    Args:
        selected: List of selected normalized feature column names.
        cfg: Configuration dictionary containing table2_columns and table2_ambiguous.

    Returns:
        dict containing overlap count, Jaccard similarity, matched, missing,
        extra features, and evaluation of ambiguous Table 2 features.
    """
    pso_cfg = cfg.get("pso", {})
    t2_cols = pso_cfg.get("table2_columns", [])
    t2_ambig = pso_cfg.get("table2_ambiguous", {})

    selected_set = set(selected)
    t2_set = set(t2_cols)

    matched = [c for c in t2_cols if c in selected_set]
    missing = [c for c in t2_cols if c not in selected_set]
    extra = [c for c in selected if c not in t2_set]

    union_set = selected_set | t2_set
    jaccard = float(len(matched) / len(union_set)) if union_set else 0.0

    ambig_results = {}
    for paper_name, candidates in t2_ambig.items():
        matched_cand = [c for c in candidates if c in selected_set]
        ambig_results[paper_name] = {
            "candidates": candidates,
            "selected_candidates": matched_cand,
            "any_selected": bool(len(matched_cand) > 0),
        }

    return {
        "overlap_count": len(matched),
        "jaccard_similarity": jaccard,
        "matched_features": matched,
        "missing_features": missing,
        "extra_features": extra,
        "ambiguous_features": ambig_results,
    }


def run_pso(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
    fitness_fn: Any = None,
    force: bool = False,
    cache_path: Path | str | None = None,
    pipeline: str = "pipeline_a",
    train_index_hash: str | None = None,
) -> dict[str, Any]:
    """Run Particle Swarm Optimization for feature selection.

    Args:
        X: Feature matrix DataFrame. Must NOT contain ID or label columns.
        y: Binary target label Series.
        cfg: Configuration dictionary.
        fitness_fn: Optional pre-constructed callable fitness function(mask) -> float.
        force: If True, bypass cache and recompute.
        cache_path: Optional custom path for reading/writing cache. Defaults to
            results/cache/pso_{pipeline}.json if caching is enabled.
        pipeline: Pipeline identifier (default "pipeline_a").
        train_index_hash: Optional SHA-256 hash of training index.

    Returns:
        dict containing:
            - selected_features: List of selected column names in original order.
            - mask: Boolean list of selected features.
            - best_rmse: Global best RMSE achieved.
            - n_selected: Number of selected features.
            - logbook: Per-iteration metrics from iteration 0 to max_iterations.
            - evaluations: Total number of fitness evaluations performed.
            - seed: Random seed used.

    Raises:
        ValueError: If any ID or label column is detected in X.
    """
    # 1. Validate inputs: ensure no identifier or label columns leak into X
    forbidden_cols = set()
    for col in cfg.get("data", {}).get("id_columns", []):
        forbidden_cols.add(normalise_column_name(col))
    raw_label = cfg.get("data", {}).get("label_column")
    if raw_label:
        forbidden_cols.add(normalise_column_name(raw_label))

    for c in X.columns:
        if normalise_column_name(c) in forbidden_cols:
            raise ValueError(
                f"Forbidden ID or label column '{c}' found in feature matrix X."
            )

    seed = int(cfg.get("seed", 42))
    pso_cfg = cfg.get("pso", {})
    use_cache = bool(pso_cfg.get("use_cache", True))

    default_cache = Path(cfg.get("paths", {}).get("results", "results/")) / "cache" / f"pso_{pipeline}.json"
    target_cache_path = Path(cache_path) if cache_path is not None else default_cache

    current_hash = compute_pso_cache_hash(
        cfg,
        (len(X), len(X.columns)),
        list(X.columns),
        seed,
        pipeline=pipeline,
        train_index_hash=train_index_hash,
    )

    # 2. Check cache
    if use_cache and not force and target_cache_path.exists():
        try:
            with open(target_cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if cached_data.get("hash") == current_hash and "result" in cached_data:
                print(f"Loaded PSO result from cache: {target_cache_path} (hash: {current_hash[:8]})")
                return cached_data["result"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            print(f"Cache read failed ({exc}); recomputing PSO.")

    # 3. Setup fitness function
    if fitness_fn is None:
        fitness_fn = make_rmse_fitness(X, y, cfg)

    # 4. Swarm parameters
    rng = np.random.default_rng(seed)
    n_particles = int(pso_cfg.get("swarm_size", 30))
    n_iterations = int(pso_cfg.get("iterations", 50))
    c1 = float(pso_cfg.get("c1", 2.0))
    c2 = float(pso_cfg.get("c2", 2.0))
    w_initial = float(pso_cfg.get("w_initial", 0.9))
    alpha = float(pso_cfg.get("inertia_decay", 0.99))
    seg = float(pso_cfg.get("seg", 10))
    bounds = tuple(float(b) for b in pso_cfg.get("bounds", [0.0, 1.0]))
    threshold = float(pso_cfg.get("select_threshold", 0.5))
    v_max = (bounds[1] - bounds[0]) / seg
    d_features = X.shape[1]

    # 5. Initialize population positions and velocities (Eq. 3-7)
    positions = rng.uniform(bounds[0], bounds[1], size=(n_particles, d_features))
    velocities = rng.uniform(-v_max, v_max, size=(n_particles, d_features))

    # 6. Evaluate initial personal bests
    pbest_pos = positions.copy()
    pbest_fitness = np.zeros(n_particles, dtype=float)
    for i in range(n_particles):
        pbest_fitness[i] = fitness_fn(positions[i] > threshold)

    # 7. Global best initialization (Algorithm 1 line 7 read as swarm-wide gbest)
    best_idx = int(np.argmin(pbest_fitness))
    gbest_pos = pbest_pos[best_idx].copy()
    gbest_fitness = float(pbest_fitness[best_idx])

    logbook: list[dict[str, Any]] = []
    logbook.append({
        "iteration": 0,
        "w": float(w_initial),
        "nevals_cumulative": int(fitness_fn.eval_count),
        "best_rmse": float(gbest_fitness),
        "mean_rmse": float(np.mean(pbest_fitness)),
        "std_rmse": float(np.std(pbest_fitness)),
        "min_rmse": float(np.min(pbest_fitness)),
        "max_rmse": float(np.max(pbest_fitness)),
        "n_selected_gbest": int(np.sum(gbest_pos > threshold)),
    })

    # 8. Main optimization loop (j = 1 to iterations)
    for j in range(1, n_iterations + 1):
        w = float(w_initial * (alpha**j))
        for i in range(n_particles):
            r1 = rng.uniform(0.0, 1.0, size=d_features)
            r2 = rng.uniform(0.0, 1.0, size=d_features)

            new_x, new_v = pso_update_particle(
                positions[i],
                velocities[i],
                pbest_pos[i],
                gbest_pos,
                w,
                c1,
                c2,
                v_max,
                bounds,
                r1,
                r2,
            )
            positions[i] = new_x
            velocities[i] = new_v

            fit = fitness_fn(positions[i] > threshold)
            if fit < pbest_fitness[i]:
                pbest_fitness[i] = fit
                pbest_pos[i] = positions[i].copy()
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest_pos = positions[i].copy()

        logbook.append({
            "iteration": j,
            "w": float(w),
            "nevals_cumulative": int(fitness_fn.eval_count),
            "best_rmse": float(gbest_fitness),
            "mean_rmse": float(np.mean(pbest_fitness)),
            "std_rmse": float(np.std(pbest_fitness)),
            "min_rmse": float(np.min(pbest_fitness)),
            "max_rmse": float(np.max(pbest_fitness)),
            "n_selected_gbest": int(np.sum(gbest_pos > threshold)),
        })

    # 9. Format return payload
    final_mask = gbest_pos > threshold
    selected_cols = [c for c, m in zip(X.columns, final_mask) if m]

    result = {
        "selected_features": selected_cols,
        "mask": [bool(m) for m in final_mask],
        "best_rmse": float(gbest_fitness),
        "n_selected": int(np.sum(final_mask)),
        "logbook": logbook,
        "evaluations": int(fitness_fn.eval_count),
        "seed": seed,
    }

    # 10. Save cache if enabled
    if use_cache and target_cache_path:
        target_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "hash": current_hash,
            "result": result,
        }
        with open(target_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f, indent=2)

    return result


if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Run PSO feature selection (Pipeline A).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of PSO, bypassing any existing cache.",
    )
    args = parser.parse_args()

    cfg_path = Path("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Load raw dataset
    t_load_start = time.time()
    df, col_mapping = load_raw(cfg)
    t_load = time.time() - t_load_start

    # 2. Split columns
    t_split_start = time.time()
    X, y, meta = split_columns(df, cfg)
    t_split = time.time() - t_split_start

    # 3. Clean missing numeric values
    t_clean_start = time.time()
    X_clean, y_clean, meta_clean, cleaning_report = clean_missing(X, y, meta, cfg)
    t_clean = time.time() - t_clean_start

    # 4. Token frequency encoding on cleaned data
    t_encode_start = time.time()
    encoder = TokenFrequencyEncoder(columns=cfg["data"]["categorical_columns"])
    X_clean_enc = encoder.fit_transform(X_clean)
    t_encode = time.time() - t_encode_start

    # 5. Fit and apply z-score scaler
    t_scale_start = time.time()
    scaler = fit_scaler(X_clean_enc)
    X_scaled = apply_scaler(scaler, X_clean_enc)
    t_scale = time.time() - t_scale_start

    # 6. Diagnostics (pre-PSO baselines)
    diag_fitness_fn = make_rmse_fitness(X_scaled, y_clean, cfg)

    # (a) Majority-class baseline (always-benign predictor)
    # y in {0, 1}, predicting 0 yields squared error y^2 = y. RMSE = sqrt(mean(y))
    rmse_majority = float(np.sqrt(np.mean(y_clean**2)))

    # (b) All features evaluated with the same CV harness and DecisionTreeClassifier
    all_features_mask = np.ones(X_scaled.shape[1], dtype=bool)
    rmse_all_features = float(diag_fitness_fn(all_features_mask))

    # (c) Table 2 feature sets evaluation
    pso_cfg = cfg.get("pso", {})
    t2_cols = pso_cfg.get("table2_columns", [])
    t2_ambig = pso_cfg.get("table2_ambiguous", {})

    t2_unambig_mask = np.array([c in t2_cols for c in X_scaled.columns], dtype=bool)
    rmse_table2_unambig = float(diag_fitness_fn(t2_unambig_mask))

    cand1_col = "erc20_uniq_sent_addr"
    cand2_col = "erc20_uniq_sent_addr_1"
    t2_cand1_mask = np.array([c in t2_cols or c == cand1_col for c in X_scaled.columns], dtype=bool)
    t2_cand2_mask = np.array([c in t2_cols or c == cand2_col for c in X_scaled.columns], dtype=bool)

    rmse_table2_cand1 = float(diag_fitness_fn(t2_cand1_mask))
    rmse_table2_cand2 = float(diag_fitness_fn(t2_cand2_mask))

    # 7. Run PSO feature selection
    t_pso_start = time.time()
    pso_result = run_pso(
        X_scaled,
        y_clean,
        cfg,
        force=args.force,
    )
    t_pso = time.time() - t_pso_start

    # 8. Compare with Table 2
    table2_comp = compare_with_table2(pso_result["selected_features"], cfg)

    t_total = time.time() - t0

    timings = {
        "load_raw": t_load,
        "split_columns": t_split,
        "clean_missing": t_clean,
        "token_frequency_encoder": t_encode,
        "fit_apply_scaler": t_scale,
        "pso": t_pso,
        "total": t_total,
    }

    diagnostics = {
        "rmse_majority": rmse_majority,
        "rmse_all_features": rmse_all_features,
        "rmse_table2_features": rmse_table2_cand1,  # Primary mapping candidate
        "rmse_table2_candidate1": rmse_table2_cand1,
        "rmse_table2_candidate2": rmse_table2_cand2,
        "rmse_table2_unambiguous_13": rmse_table2_unambig,
        "paper_reported_rmse": 0.3443,
        "paper_feature_count": 14,
    }

    payload = {
        "selected_features": pso_result["selected_features"],
        "best_rmse": pso_result["best_rmse"],
        "n_selected": pso_result["n_selected"],
        "evaluations": pso_result["evaluations"],
        "logbook": pso_result["logbook"],
        "compare_with_table2": table2_comp,
        "diagnostics": diagnostics,
    }

    out_file = write_results(
        phase="phase1_4",
        name="pso",
        payload=payload,
        cfg=cfg,
        timings=timings,
    )

    print("\n=== PSO Feature Selection Completed ===")
    print(f"Results written to: {out_file}")
    print(f"Features selected: {pso_result['n_selected']} (Paper: 14)")
    print(f"Best RMSE: {pso_result['best_rmse']:.4f} (Paper: 0.3443)")
    print(f"Majority baseline RMSE: {rmse_majority:.4f}")
    print(f"All 47 features RMSE: {rmse_all_features:.4f}")
    print(f"Table 2 (14 features, cand1) RMSE: {rmse_table2_cand1:.4f}")
    print(f"Table 2 (14 features, cand2) RMSE: {rmse_table2_cand2:.4f}")
    print(f"Table 2 overlap: {table2_comp['overlap_count']} / 13 unambiguous features (Jaccard: {table2_comp['jaccard_similarity']:.4f})")
    print(f"Total fitness evaluations: {pso_result['evaluations']}")
    print(f"PSO execution time: {t_pso:.2f}s (Total pipeline: {t_total:.2f}s)")
