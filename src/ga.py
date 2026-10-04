"""Genetic Algorithm (GA) hyperparameter optimization module for Pipeline A.

Reference:
- El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts
  in the Ethereum Blockchain Network", Cryptography 2025, 9, 63.
  Section 4.3 (Classification Stage: GA Hyperparameter Optimization)
  Algorithm 2 (GA workflow for hyperparameter tuning)
  Figure 7 (GA workflow diagram)
  Tables 1, 3, 4 (GA statistics and convergence over 20 generations)
  Tables 5-7 (Hyperparameter tuning for XGBoost, SVM, and Isolation Forest)
  Table 9 (Model evaluation metrics after GA tuning)
  Algorithm 6 (Full Pipeline A order: clean -> z-score -> PSO -> SMOTE -> split -> GA -> models)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from deap import algorithms, base, creator, tools
from sklearn.model_selection import StratifiedKFold
from xgboost import __version__ as xgboost_version

from src.evaluate import check_consistency, compute_metrics, write_results
from src.models import (
    build_isolation_forest,
    build_svm,
    build_xgboost,
    fit_predict_scores,
)
from src.pipeline_a import build_pipeline_a_data

# Ensure DEAP creator classes are registered once safely
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)


def clip_individual(
    individual: list[float],
    param_specs: list[dict[str, Any]],
) -> list[float]:
    """Clip individual gene values to the parameter search space bounds.

    Args:
        individual: List of continuous gene values.
        param_specs: List of parameter specification dictionaries.

    Returns:
        The modified individual with all genes clamped within bounds.
    """
    for i, spec in enumerate(param_specs):
        low, high = spec["bounds"]
        individual[i] = float(min(max(individual[i], low), high))
    return individual


def decode_individual(
    individual: list[float],
    param_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Decode individual gene values into a dictionary of hyperparameter values.

    Handles:
    - 'int': rounded to nearest integer within [low, high].
    - 'float': continuous float within [low, high].
    - 'log10_float': decoded as 10**gene, where gene is in [low, high].

    Args:
        individual: List of continuous gene values.
        param_specs: List of parameter specification dictionaries.

    Returns:
        dict of decoded hyperparameter names and values.
    """
    params: dict[str, Any] = {}
    for val, spec in zip(individual, param_specs):
        name = spec["name"]
        ptype = spec["type"]
        low, high = spec["bounds"]
        clipped = float(min(max(val, low), high))

        if ptype == "int":
            params[name] = round(clipped)
        elif ptype == "log10_float":
            params[name] = float(10.0**clipped)
        elif ptype == "float":
            params[name] = float(clipped)
        else:
            raise ValueError(f"Unknown parameter type '{ptype}' for parameter '{name}'.")

    return params


class GAFitness:
    """Stratified K-fold cross-validated fitness evaluator on the train split only."""

    def __init__(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_specs: list[dict[str, Any]],
        cfg: dict[str, Any],
    ) -> None:
        self.model_name = model_name.strip().lower()
        self.param_specs = param_specs
        self.cfg = cfg
        self.eval_count = 0

        ga_cfg = cfg.get("ga", {})
        self.cv_folds = int(ga_cfg.get("cv_folds", 3))
        self.seed = int(cfg.get("seed", 42))

        self.X_arr = np.asarray(X_train, dtype=float)
        self.y_arr = np.asarray(y_train, dtype=int)
        self.columns = list(X_train.columns)

        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.seed,
        )
        self.splits = list(skf.split(self.X_arr, self.y_arr))

    def _build_model(self, params: dict[str, Any]) -> Any:
        if self.model_name in {"xgboost", "xgb"}:
            return build_xgboost(params, self.cfg)
        elif self.model_name in {"svm", "svc"}:
            return build_svm(params, self.cfg)
        elif self.model_name in {"isolation_forest", "if", "isoforest"}:
            return build_isolation_forest(params, self.cfg)
        else:
            raise ValueError(f"Unsupported model name '{self.model_name}'.")

    def __call__(self, individual: list[float]) -> tuple[float]:
        """Evaluate stratified cross-validation mean accuracy.

        Args:
            individual: List of continuous genes.

        Returns:
            tuple of (mean_accuracy,).
        """
        self.eval_count += 1
        params = decode_individual(individual, self.param_specs)

        fold_accuracies: list[float] = []
        for train_idx, val_idx in self.splits:
            X_tr = pd.DataFrame(self.X_arr[train_idx], columns=self.columns)
            y_tr = pd.Series(self.y_arr[train_idx])
            X_val = pd.DataFrame(self.X_arr[val_idx], columns=self.columns)
            y_val = self.y_arr[val_idx]

            model = self._build_model(params)
            y_pred, _ = fit_predict_scores(self.model_name, model, X_tr, y_tr, X_val)
            acc = float(np.mean(y_pred == y_val))
            fold_accuracies.append(acc)

        mean_acc = float(np.mean(fold_accuracies))
        if mean_acc <= 0.0:
            raise ValueError(
                f"Fitness must be positive for roulette selection; got {mean_acc}."
            )

        return (mean_acc,)


def make_ga_fitness(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_specs: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> GAFitness:
    """Factory creating a GAFitness callable.

    Takes X_train, y_train and cfg arguments (no data loading inside)
    so Phase 2.0 can reuse it directly.

    Args:
        model_name: Identifier for model ('xgboost', 'svm', 'isolation_forest').
        X_train: Training feature DataFrame.
        y_train: Training label Series.
        param_specs: Parameter search space specifications.
        cfg: Configuration dictionary.

    Returns:
        GAFitness instance.
    """
    return GAFitness(model_name, X_train, y_train, param_specs, cfg)


def compute_ga_cache_hash(
    model_name: str,
    feature_set: str,
    ga_cfg: dict[str, Any],
    param_specs: list[dict[str, Any]],
    X_train_shape: tuple[int, int],
    y_train_shape: tuple[int, ...],
    feature_names: list[str],
    seed: int,
    library_versions: dict[str, str],
) -> str:
    """Compute deterministic SHA-256 hash for GA cache validation.

    Args:
        model_name: Identifier for model.
        feature_set: Feature set name ('pso' or 'table2').
        ga_cfg: Genetic algorithm configuration sub-dictionary.
        param_specs: Search space parameter specs.
        X_train_shape: Dimensions of training feature matrix.
        y_train_shape: Dimensions of training target vector.
        feature_names: Names of feature columns in training data.
        seed: Random seed.
        library_versions: Pinned library version mapping.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    payload = {
        "model_name": model_name,
        "feature_set": feature_set,
        "ga_cfg": ga_cfg,
        "param_specs": param_specs,
        "X_train_shape": list(X_train_shape),
        "y_train_shape": list(y_train_shape),
        "feature_names": list(feature_names),
        "seed": seed,
        "library_versions": library_versions,
    }
    dumped = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def run_ga(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
    param_specs: list[dict[str, Any]] | None = None,
    feature_set: str = "pso",
    force: bool = False,
    cache_path: Path | str | None = None,
    pop_size_override: int | None = None,
    n_gen_override: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute Genetic Algorithm hyperparameter optimization with eaSimple semantics.

    Args:
        model_name: Identifier ('xgboost', 'svm', or 'isolation_forest').
        X_train: Training feature DataFrame (train split only).
        y_train: Training label Series (train split only).
        cfg: Configuration dictionary.
        param_specs: Search space parameter specs (defaults to cfg search_spaces).
        feature_set: Feature set identifier ('pso' or 'table2').
        force: If True, bypass cache and recompute.
        cache_path: Optional explicit cache file path.
        pop_size_override: Optional population size override (used for budget probe).
        n_gen_override: Optional generation count override (used for budget probe).
        verbose: If True, print generational progress.

    Returns:
        dict containing best_params, best_fitness, best_individual,
        evaluations, logbook, elapsed_seconds, and cache metadata.
    """
    t_start = time.time()
    ga_cfg = cfg.get("ga", {})
    seed = int(cfg.get("seed", 42))

    if param_specs is None:
        search_spaces = ga_cfg.get("search_spaces", {})
        if model_name not in search_spaces:
            raise KeyError(f"No search space found in config for model '{model_name}'.")
        param_specs = search_spaces[model_name]

    pop_size = pop_size_override if pop_size_override is not None else int(ga_cfg.get("population", 50))
    n_gen = n_gen_override if n_gen_override is not None else int(ga_cfg.get("generations", 20))
    cxpb = float(ga_cfg.get("cxpb", 0.5))
    mutpb = float(ga_cfg.get("mutpb", 0.2))
    indpb = float(ga_cfg.get("indpb", 0.2))
    sigma_frac = float(ga_cfg.get("sigma_fraction", 0.1))
    blend_alpha = float(ga_cfg.get("blend_alpha", 0.5))
    use_cache = bool(ga_cfg.get("use_cache", True)) and (pop_size_override is None) and (n_gen_override is None)

    lib_versions = {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost_version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    current_hash = compute_ga_cache_hash(
        model_name=model_name,
        feature_set=feature_set,
        ga_cfg=ga_cfg,
        param_specs=param_specs,
        X_train_shape=X_train.shape,
        y_train_shape=y_train.shape,
        feature_names=list(X_train.columns),
        seed=seed,
        library_versions=lib_versions,
    )

    if cache_path is None:
        target_cache_path = Path(f"results/cache/ga_pipeline_a_{model_name}_{feature_set}.json")
    else:
        target_cache_path = Path(cache_path)

    # 1. Check Cache
    if use_cache and not force and target_cache_path.exists():
        with open(target_cache_path, "r", encoding="utf-8") as f:
            cache_payload = json.load(f)
        if cache_payload.get("hash") == current_hash:
            print(
                f"[GA CACHE] Reusing cached GA results for {model_name} ({feature_set}) "
                f"from {target_cache_path} (hash: {current_hash[:8]}...)"
            )
            res = cache_payload["result"]
            res["cached"] = True
            return res

    # 2. Setup Seeding and DEAP
    random.seed(seed)
    np.random.seed(seed)

    fitness_fn = make_ga_fitness(model_name, X_train, y_train, param_specs, cfg)

    bounds = [spec["bounds"] for spec in param_specs]
    sigmas = [sigma_frac * (b[1] - b[0]) for b in bounds]

    def create_individual() -> Any:
        genes = [random.uniform(b[0], b[1]) for b in bounds]
        return creator.Individual(genes)

    def mate_clipped(ind1: Any, ind2: Any) -> tuple[Any, Any]:
        tools.cxBlend(ind1, ind2, alpha=blend_alpha)
        clip_individual(ind1, param_specs)
        clip_individual(ind2, param_specs)
        return ind1, ind2

    def mutate_clipped(ind: Any) -> tuple[Any]:
        tools.mutGaussian(ind, mu=0.0, sigma=sigmas, indpb=indpb)
        clip_individual(ind, param_specs)
        return (ind,)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate_clipped)
    toolbox.register("mutate", mutate_clipped)
    toolbox.register("select", tools.selRoulette)
    toolbox.register("evaluate", fitness_fn)

    # 3. Setup Stats and Hall of Fame
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda vals: float(np.mean(vals)))
    stats.register("std", lambda vals: float(np.std(vals)))
    stats.register("min", lambda vals: float(np.min(vals)))
    stats.register("max", lambda vals: float(np.max(vals)))
    hof = tools.HallOfFame(1)

    # 4. Run eaSimple
    pop = toolbox.population(n=pop_size)
    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    t_end = time.time()
    elapsed = t_end - t_start

    # Format logbook records
    formatted_logbook: list[dict[str, Any]] = []
    for entry in logbook:
        formatted_logbook.append({
            "gen": int(entry["gen"]),
            "nevals": int(entry["nevals"]),
            "avg": float(entry["avg"]),
            "std": float(entry["std"]),
            "min": float(entry["min"]),
            "max": float(entry["max"]),
        })

    best_ind = [float(x) for x in hof[0]]
    best_fitness = float(hof[0].fitness.values[0])
    best_params = decode_individual(best_ind, param_specs)

    result = {
        "model_name": model_name,
        "feature_set": feature_set,
        "best_params": best_params,
        "best_fitness": best_fitness,
        "best_individual": best_ind,
        "evaluations": int(fitness_fn.eval_count),
        "logbook": formatted_logbook,
        "elapsed_seconds": elapsed,
        "seed": seed,
        "cached": False,
    }

    # 5. Save Cache if enabled
    if use_cache and target_cache_path:
        target_cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "hash": current_hash,
            "result": result,
        }
        with open(target_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_payload, f, indent=2)

    return result


def run_budget_probe(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
    feature_set: str = "pso",
) -> dict[str, Any]:
    """Execute lightweight budget probe (population 4, 1 generation) to measure evaluation cost.

    Args:
        model_name: Identifier ('xgboost', 'svm', or 'isolation_forest').
        X_train: Training feature DataFrame.
        y_train: Training label Series.
        cfg: Configuration dictionary.
        feature_set: Feature set identifier ('pso' or 'table2').

    Returns:
        dict containing measured seconds per eval and projections for 1000 and 600 evals.
    """
    t0 = time.time()
    probe_res = run_ga(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        cfg=cfg,
        feature_set=feature_set,
        force=True,
        pop_size_override=4,
        n_gen_override=1,
        verbose=False,
    )
    t1 = time.time()
    elapsed = t1 - t0
    n_evals = probe_res["evaluations"]
    sec_per_eval = elapsed / n_evals if n_evals > 0 else 0.0

    est_1000 = 1000.0 * sec_per_eval
    est_600 = 600.0 * sec_per_eval

    return {
        "model": model_name,
        "feature_set": feature_set,
        "probe_pop": 4,
        "probe_gen": 1,
        "evaluations_measured": n_evals,
        "elapsed_seconds": elapsed,
        "sec_per_eval": sec_per_eval,
        "est_1000_seconds": est_1000,
        "est_1000_minutes": est_1000 / 60.0,
        "est_600_seconds": est_600,
        "est_600_minutes": est_600 / 60.0,
        "under_30_min": est_600 < 1800.0,
    }


def find_latest_phase1_5_results(results_dir: Path | str = "results") -> dict[str, Any] | None:
    """Find and load the latest Phase 1.5 defaults results JSON."""
    dir_path = Path(results_dir)
    matches = sorted(dir_path.glob("phase1_5_pipeline_a_defaults_*.json"))
    if not matches:
        return None
    latest_file = matches[-1]
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_ga_on_test(
    model_name: str,
    best_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict[str, Any],
    feature_set: str = "pso",
) -> dict[str, Any]:
    """Train the model with tuned hyperparameters on full train split and evaluate on test set.

    Compares resulting test metrics with:
    - Phase 1.5 default baseline metrics
    - Paper Table 9 GA metrics
    - Paper Tables 5-7 tuned hyperparameters

    Args:
        model_name: Identifier ('xgboost', 'svm', 'isolation_forest').
        best_params: Tuned hyperparameters from GA.
        X_train: Full train split features.
        y_train: Full train split labels.
        X_test: Test split features.
        y_test: Test split labels.
        cfg: Configuration dictionary.
        feature_set: Feature set identifier ('pso' or 'table2').

    Returns:
        dict containing metrics, default comparison, paper comparison, and consistency checks.
    """
    t_start = time.time()

    # 1. Build and fit tuned model
    norm_name = model_name.strip().lower()
    if norm_name in {"xgboost", "xgb"}:
        model = build_xgboost(best_params, cfg)
    elif norm_name in {"svm", "svc"}:
        model = build_svm(best_params, cfg)
    elif norm_name in {"isolation_forest", "if", "isoforest"}:
        model = build_isolation_forest(best_params, cfg)
    else:
        raise ValueError(f"Unsupported model name '{model_name}'.")

    y_pred, scores = fit_predict_scores(model_name, model, X_train, y_train, X_test)
    fit_predict_time = time.time() - t_start

    # 2. Compute test metrics
    metrics = compute_metrics(y_test, y_pred, scores)

    # 3. Compare with Phase 1.5 defaults
    defaults_data = find_latest_phase1_5_results()
    default_comparison: dict[str, Any] = {}
    if defaults_data and "results" in defaults_data:
        feat_results = defaults_data["results"].get(feature_set, {})
        model_defaults = feat_results.get("models", {}).get(model_name, {}).get("metrics", {})
        if model_defaults:
            default_comparison = {
                "default_metrics": {
                    "accuracy": model_defaults.get("accuracy"),
                    "precision": model_defaults.get("precision"),
                    "recall": model_defaults.get("recall"),
                    "f1": model_defaults.get("f1"),
                    "roc_auc": model_defaults.get("roc_auc"),
                    "mae": model_defaults.get("mae"),
                },
                "deltas_to_default": {
                    "accuracy": float(metrics["accuracy"] - model_defaults.get("accuracy", 0.0)),
                    "precision": float(metrics["precision"] - model_defaults.get("precision", 0.0)),
                    "recall": float(metrics["recall"] - model_defaults.get("recall", 0.0)),
                    "f1": float(metrics["f1"] - model_defaults.get("f1", 0.0)),
                    "roc_auc": float(metrics["roc_auc"] - model_defaults.get("roc_auc", 0.0)),
                    "mae": float(metrics["mae"] - model_defaults.get("mae", 0.0)),
                },
            }

    # 4. Compare with Paper Table 9 and Tables 5-7
    paper_ga_metrics = cfg.get("paper_results", {}).get("ga", {}).get(model_name, {})
    paper_ga_params = cfg.get("paper_results", {}).get("ga_params", {}).get(model_name, {})

    paper_comparison: dict[str, Any] = {
        "paper_reported_metrics": paper_ga_metrics,
        "deltas_to_paper_metrics": {
            "accuracy": float(metrics["accuracy"] - paper_ga_metrics.get("accuracy", 0.0)) if "accuracy" in paper_ga_metrics else None,
            "precision": float(metrics["precision"] - paper_ga_metrics.get("precision", 0.0)) if "precision" in paper_ga_metrics else None,
            "recall": float(metrics["recall"] - paper_ga_metrics.get("recall", 0.0)) if "recall" in paper_ga_metrics else None,
            "f1": float(metrics["f1"] - paper_ga_metrics.get("f1", 0.0)) if "f1" in paper_ga_metrics else None,
            "auc": float(metrics["roc_auc"] - paper_ga_metrics.get("auc", 0.0)) if "auc" in paper_ga_metrics else None,
            "mae": float(metrics["mae"] - paper_ga_metrics.get("mae", 0.0)) if "mae" in paper_ga_metrics else None,
        },
        "paper_reported_params": paper_ga_params,
        "our_tuned_params": best_params,
    }

    # 5. Consistency check
    expected_counts = {
        "n_test": len(y_test),
        0: int((y_test == 0).sum()),
        1: int((y_test == 1).sum()),
    }
    consistency = check_consistency({model_name: metrics}, expected_counts=expected_counts)

    return {
        "metrics": metrics,
        "fit_predict_time_seconds": fit_predict_time,
        "default_comparison": default_comparison,
        "paper_comparison": paper_comparison,
        "consistency_checks": consistency,
    }


def write_logbook_csv(logbook: list[dict[str, Any]], out_path: Path | str) -> None:
    """Write GA logbook records to CSV matching paper Table 4 columns.

    Columns: gen, nevals, avg, std, min, max.

    Args:
        logbook: List of generation dictionary records.
        out_path: Output CSV file path.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["gen", "nevals", "avg", "std", "min", "max"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in logbook:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    """CLI entrypoint for Phase 1.6 GA hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Run GA hyperparameter tuning (Pipeline A).")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgboost", "svm", "isolation_forest"],
        help="Model to optimize with GA.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="pso",
        choices=["pso", "table2"],
        help="Feature set to extract and evaluate ('pso' or 'table2').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of GA, bypassing any existing cache.",
    )
    parser.add_argument(
        "--budget-probe",
        action="store_true",
        help="Run lightweight probe (population 4, 1 gen) to estimate runtime.",
    )
    args = parser.parse_args()

    cfg_path = Path("config/config.yaml")
    if not cfg_path.exists():
        sys.exit("Error: config/config.yaml not found.")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Prepare Pipeline A data
    print(f"Preparing Pipeline A data (feature_set={args.feature_set})...")
    data = build_pipeline_a_data(cfg, feature_set=args.feature_set)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(
        f"Data prepared: X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_test={X_test.shape}, y_test={y_test.shape}"
    )

    # 2. Budget Probe Mode
    if args.budget_probe:
        print(f"\n--- Running Budget Probe for {args.model} ---")
        probe_res = run_budget_probe(args.model, X_train, y_train, cfg, feature_set=args.feature_set)
        print(f"Model:                {probe_res['model']}")
        print(f"Evaluations Measured: {probe_res['evaluations_measured']} (pop={probe_res['probe_pop']}, gen={probe_res['probe_gen']})")
        print(f"Elapsed Time:         {probe_res['elapsed_seconds']:.2f}s")
        print(f"Seconds per Eval:     {probe_res['sec_per_eval']:.4f}s")
        print(f"Est. 1000 evals (50x20): {probe_res['est_1000_seconds']:.1f}s ({probe_res['est_1000_minutes']:.2f} min)")
        print(f"Est. ~600 evals (Table 4): {probe_res['est_600_seconds']:.1f}s ({probe_res['est_600_minutes']:.2f} min)")
        print(f"Under 30 minutes:     {probe_res['under_30_min']}")
        return

    # 3. Full GA Optimization
    print(f"\n--- Running Full GA for {args.model} ---")
    t0 = time.time()
    ga_res = run_ga(
        model_name=args.model,
        X_train=X_train,
        y_train=y_train,
        cfg=cfg,
        feature_set=args.feature_set,
        force=args.force,
        verbose=True,
    )
    ga_elapsed = time.time() - t0

    best_params = ga_res["best_params"]
    best_fitness = ga_res["best_fitness"]
    n_evals = ga_res["evaluations"]
    print(f"\nGA Finished! Total evaluations: {n_evals}, Best CV Accuracy: {best_fitness:.5f}")
    print(f"Best hyperparameters: {json.dumps(best_params, indent=2)}")

    # 4. Evaluate on Test Set
    print("\nEvaluating tuned model on test set...")
    eval_res = evaluate_ga_on_test(
        model_name=args.model,
        best_params=best_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cfg=cfg,
        feature_set=args.feature_set,
    )

    test_metrics = eval_res["metrics"]
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{np.array(test_metrics['confusion_matrix'])}")

    # 5. Write Outputs
    timings = {
        "ga_optimization": ga_elapsed,
        "test_fit_predict": eval_res["fit_predict_time_seconds"],
    }
    out_payload: dict[str, Any] = {
        "model": args.model,
        "feature_set": args.feature_set,
        "search_space": cfg.get("ga", {}).get("search_spaces", {}).get(args.model, []),
        "best_individual": ga_res["best_individual"],
        "best_params": best_params,
        "best_fitness_cv_accuracy": best_fitness,
        "total_evaluations": n_evals,
        "logbook": ga_res["logbook"],
        "test_metrics": test_metrics,
        "default_comparison": eval_res["default_comparison"],
        "paper_comparison": eval_res["paper_comparison"],
        "consistency_checks": eval_res["consistency_checks"],
    }

    out_json_path = write_results("phase1_6", f"ga_{args.model}", out_payload, cfg, timings)
    print(f"Saved full results JSON: {out_json_path}")

    logbook_csv_path = out_json_path.with_name(out_json_path.stem + "_logbook.csv")
    write_logbook_csv(ga_res["logbook"], logbook_csv_path)
    print(f"Saved logbook CSV: {logbook_csv_path}")


if __name__ == "__main__":
    main()
