# Ethereum Suspicious-Account Detection

Reproduction, benchmark and extension of El-Attar et al., *"An Optimized Framework for Detecting Suspicious Accounts in the Ethereum Blockchain Network"*, Cryptography 2025, 9, 63. The framework combines PSO feature selection, GA hyperparameter tuning and XGBoost/SVM/Isolation Forest classifiers. This repo adds a leakage-safe pipeline, speed/accuracy optimisation, tests of the paper's untested claims, and an on-chain risk registry (Solidity + Hardhat).

To our knowledge, no open-source implementation of the complete framework exists. This is a replication, not a new method.

**Status:** Phase 2.0 complete; 2.1 onwards not started.

## Results

| Model | Stage | Paper | Pipeline A (replication) | Pipeline B (leakage-safe) | Optimised |
|---|---|---|---|---|---|
| XGBoost | default | 0.975 | 0.9967 | 0.9945 | not yet run |
| XGBoost | GA | 0.992 | 0.9971 | 0.9911 | not yet run |
| SVM | default | 0.744 | 0.9377 | 0.9196 | not yet run |
| SVM | GA | 0.87 | 0.9883 | 0.9795 | not yet run |
| Isolation Forest | default | 0.694 | 0.4395 | 0.7399 | not yet run |
| Isolation Forest | GA | 0.824 | 0.4897 | 0.3172 | not yet run |

Test accuracy shown on the 22 PSO-selected features. Source: `results/phase1_8_benchmark_20261004_044733.json` and `results/phase2_0d_ladder_benchmark_*.json` (and `.md`).
- **Evaluation splits and comparability:** Pipeline A accuracy is evaluated on an unstratified, artificially balanced test set containing 40.62% synthetic accounts (1,245 synthetic / 3,065 total test rows). In contrast, Pipeline B (and Rung L1) evaluates on a strictly natural-ratio test set (1,803 accounts: 1,533 benign, 270 suspicious; 0 synthetic rows), where always predicting benign yields **85.02% majority-class baseline accuracy**. Due to synthetic balancing in Pipeline A, accuracy and precision are not directly comparable across Pipeline A and Pipeline B; ROC-AUC and PR-AUC provide the closest methodological comparison.
- **Below-chance anomaly detection:** Isolation Forest achieves test accuracy of 0.4395 (default) and 0.4897 (GA) on Pipeline A, and 0.7399 (default) and 0.3172 (GA) on Pipeline B, falling below the majority baseline; its ROC-AUC is well below 0.5 (0.1356 default, 0.1957 GA on Pipeline B), indicating an inverted anomaly ranking on this feature distribution.

### Replication caveats

Pipeline A is a best-effort replication, not an exact one. Numbers differ from the paper for reasons we could only partly isolate:

- **Dataset:** the data we used compared with the paper's description (9,841 accounts; 49 features; raw 7,662 benign, 2,179 suspicious vs paper 7,663 / 2,178; post-cleaning 7,662 / 1,350 vs paper 7,662 / 1,328). We could not obtain anything closer.
- **Feature subset:** our PSO selected 22 features, not the paper's 14 (Table 2). The paper does not specify swarm size, c1, c2, SEG, or how continuous positions map to a feature subset, so these are our choices (see `docs/ASSUMPTIONS.md`).
- **Other unspecified settings:** GA crossover/mutation rates, SMOTE settings, SVM kernel, XGBoost subsample, categorical encoding and seeds are also assumptions.
- **Pipeline order:** following the paper's numbers, SMOTE runs before the split in Pipeline A, so 40.62% of the Pipeline A test set is synthetic. Phase 2.0 implements the leakage-safe Pipeline B (split first, fold-wise SMOTE, test natural), establishing honest baselines.

Absolute scores should not be compared directly with the paper due to these differences. See `docs/FINDINGS.md` for the full list of inconsistencies.

## Setup

Requires Python 3.14

**macOS:** XGBoost needs the OpenMP runtime: `brew install libomp`.

```bash
git clone https://github.com/rohpod/ethereum-suspicious-account-detection.git
cd ethereum-suspicious-account-detection
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Data:** not included. Place the dataset at `data/transaction_dataset.csv`. Download the Kaggle dataset linked in the paper's Data Availability Statement (see `docs/ASSUMPTIONS.md` for status and licence notes).

## Reproduction

Execute the 10-step reproduction sequence in order:

```bash
# 1. Validate dataset shape, feature columns, and missingness (~0.09s)
python -m src.data

# 2. Clean numeric NaNs and apply z-score standardisation (~0.09s)
python -m src.preprocess

# 3. Verify paper confusion matrices and metrics formulas (~0.003s)
python -m src.evaluate

# 4. Run Particle Swarm Optimization feature selection (~39.9s uncached; cached in results/cache/pso_pipeline_a.json)
python -m src.pso

# 5. Execute Pipeline A SMOTE, 80/20 split, and evaluate default models (~5.2s)
python -m src.pipeline_a

# 6. GA hyperparameter tuning for XGBoost (~212.9s / ~3.5 min; cached in results/cache/ga_pipeline_a_xgboost_pso.json)
python -m src.ga --model xgboost

# 7. GA hyperparameter tuning for Isolation Forest (~364.7s / ~6.1 min; cached in results/cache/ga_pipeline_a_isolation_forest_pso.json)
python -m src.ga --model isolation_forest

# 8. GA hyperparameter tuning for SVM (~1366.3s / ~22.8 min; cached in results/cache/ga_pipeline_a_svm_pso.json)
python -m src.ga --model svm

# 9. Evaluate CART and LOF comparison baselines (~3.8s)
python -m src.baselines

# 10. Generate master benchmark table and cross-model checks (~0.08s)
python -m src.benchmark
```

- **Caching:** PSO and GA results are cached in `results/cache/` keyed by a deterministic configuration/data SHA-256 hash and reused only on a matching hash (pass `--force` to recompute).
- **Tests:** Run unit tests with `pytest -q`. `RUN_SLOW_TESTS=1 pytest -q` runs the slow real-data test.

## Repo map

- `config/config.yaml` — seeds, paths, PSO/GA budgets
- `data/` — gitignored
- `src/` — data, preprocess, pso, ga, models, evaluate, pipelines, bridge
- `notebooks/` — EDA only
- `results/` — one CSV/JSON per run
- `contracts/` — Hardhat project (Phase 4)
- `docs/` — FINDINGS, ASSUMPTIONS, QUESTIONS
- `tests/` — pytest
- `PLAN.md` — project plan and rules

## SDG mapping

SDG 16, target 16.4 (reducing illicit financial flows). To be confirmed with the project guide.

## Reference

El-Attar et al., Cryptography 2025, 9, 63.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Development of this application was supported by Antigravity.

