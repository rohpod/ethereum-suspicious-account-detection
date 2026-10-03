# Findings

<!-- template: ## YYYY-MM-DD: title / What / Why / Number / Surprise -->

## 2026-10-04: Phase 1.1 Data loading, feature audit, and paper replication validation

- **What:** Executed `python -m src.data` loading `data/transaction_dataset.csv` with configuration in `config/config.yaml`; output saved to `results/phase1_1_data_validation_20261003_203426.json`.
- **Why:** Verify raw dataset properties against claims in El-Attar et al. (Cryptography 2025, 9, 63 Section 4.1 & Table A2) before implementing downstream cleaning and modeling.
- **Number:**
  - Raw shape: 9,841 rows, 51 columns.
  - Class distribution: 7,662 benign (0), 2,179 suspicious (1).
  - Feature count: 47 candidate features (45 numeric, 2 categorical token types); 3 identifier columns (`unnamed_0`, `index`, `address`); 1 label (`flag`).
  - Columns minus Address and FLAG: 49. (The idea that the paper's reported 49 features derives from `51 - Address - FLAG` remains an unconfirmed hypothesis).
  - Missingness: 22,635 missing cells across 25 columns. 2,720 rows with any missing value (1,891 benign, 829 suspicious).
  - Label-linked missingness: Exactly 829 rows have missing values across all 23 numeric ERC20 features, and 100% of these rows (829/829) belong to the suspicious class (FLAG = 1). Conversely, the 1,891 benign rows with NaNs have missing values exclusively in the two categorical token columns.
  - Categorical token columns: `'erc20_most_sent_token_type'` (2,697 NaNs, 1,191 whitespace-only, 4,399 '0's) and `'erc20_most_rec_token_type'` (871 NaNs, 21 whitespace-only, 4,399 '0's).
  - Duplicate addresses: 25 duplicate addresses (50 rows total, all with FLAG = 0; 0 cross-label conflicts).
  - Duplicate feature vectors: 546 duplicate feature+label vectors (excluding ID columns).
  - Constant columns: 7 features have zero variance (all 0.0 across all non-null entries).
  - Timestamps: No absolute timestamps or block numbers exist; all time features measure relative intervals/durations in minutes.
  - Wall-clock runtime: 0.085s total (load 0.033s, split 0.004s, validate 0.034s, token encode 0.013s).
- **Surprise:**
  1. Label distribution off-by-one: Paper Section 4.1 states 7,663 benign and 2,178 suspicious accounts. The raw dataset contains 7,662 benign and 2,179 suspicious accounts. Notably, paper Section 4.2.4 gives the post-cleaning benign count as 7,662, which exactly equals the raw CSV benign count.
  2. Missing data drops almost exclusively suspicious accounts: Dropping numeric ERC20 missing values removes 829 suspicious accounts and 0 benign accounts.
  3. Duplicate addresses exist in raw data: 25 address duplicates exist despite the paper's claim in Section 4.1 that duplicates were removed.
