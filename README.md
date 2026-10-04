# Ethereum Suspicious-Account Detection

Reproduction, benchmark and extension of El-Attar et al., *"An Optimized Framework for Detecting Suspicious Accounts in the Ethereum Blockchain Network"*, Cryptography 2025, 9, 63. The framework combines PSO feature selection, GA hyperparameter tuning and XGBoost/SVM/Isolation Forest classifiers. This repo adds a leakage-safe pipeline, speed/accuracy optimisation, tests of the paper's untested claims, and an on-chain risk registry (Solidity + Hardhat).

To our knowledge, no open-source implementation of the complete framework exists. This is a replication, not a new method.

**Status:** work in progress.

## Results

| Model | Stage | Paper | Pipeline A (replication) | Pipeline B (leakage-safe) | Optimised |
|---|---|---|---|---|---|
| XGBoost | default | 0.975 | not yet run | not yet run | not yet run |
| XGBoost | GA | 0.992 | not yet run | not yet run | not yet run |
| SVM | default | 0.744 | not yet run | not yet run | not yet run |
| SVM | GA | 0.87 | not yet run | not yet run | not yet run |
| Isolation Forest | default | 0.694 | not yet run | not yet run | not yet run |
| Isolation Forest | GA | 0.824 | not yet run | not yet run | not yet run |

Accuracy shown. Every number here comes from a file in `results/`.

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

**Data:** not included. Download the Kaggle dataset linked in the paper's Data Availability Statement (see `docs/ASSUMPTIONS.md` for status and licence notes) into `data/`.

## Reproduction

Not yet available. Steps will be added as each phase lands. All seeds, paths and budgets live in `config/config.yaml`.

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

