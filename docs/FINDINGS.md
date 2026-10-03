# Findings

<!-- template: ## YYYY-MM-DD: title / What / Why / Number / Surprise -->

## 2026-10-04: Phase 1.1 Data loading, feature audit, and paper replication validation

- **What:** Executed `python -m src.data` loading `data/transaction_dataset.csv` with configuration in `config/config.yaml`; output saved to `results/phase1_1_data_validation_20261003_204929.json`.
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

## 2026-10-04: Phase 1.2 Cleaning and scaling (Pipeline A)

- **What:** Executed `python -m src.preprocess` with config in `config/config.yaml`; output saved to `results/phase1_2_cleaning_scaling_20261003_211147.json`.
- **Why:** Implement and verify data cleaning (dropping numeric NaNs) and z-score standardization according to paper Sections 4.2.1, 4.2.2, Figure 3, and Algorithm 6.
- **Number:**
  - Raw: 9,841 rows (7,662 benign, 2,179 suspicious).
  - Cleaned: 9,012 rows (7,662 benign, 1,350 suspicious), dropping 829 rows.
  - Target comparison: Benign count matches paper Section 4.2.4 (7,662) exactly ($\Delta = 0$). Cleaned suspicious count is 1,350 vs paper's 1,328 ($\Delta = +22$). Dropped rows count is 829 vs paper's 851 ($\Delta = -22$).
  - Missingness structure: Exactly 829 rows dropped, all belonging to class 1 (suspicious). Missingness in the 23 numeric ERC20 features is all-or-nothing (every dropped row has all 23 numeric ERC20 columns NaN). Including the 2 token columns, this confirms the paper's count of "25 features" with missing data.
  - Label proxy: Dropping numeric NaNs removes exclusively suspicious accounts (100% suspicious, 0% benign), confirming that missingness is heavily label-linked (critical context for Phase 3.1).
  - Naive dropna impact: Naive `df.dropna()` drops 2,720 rows leaving only 7,121 rows (5,771 benign, 1,350 suspicious), erroneously removing 1,891 benign accounts due to missing token names.
  - Duplicates: On raw data, duplicate feature vectors by class: all 47 features = 523 suspicious, 23 benign; numeric 45 features = 523 suspicious, 30 benign. On cleaned data: all 47 features = 243 suspicious, 23 benign; numeric 45 features = 243 suspicious, 30 benign (important note for Phase 2.0).
  - Scaler check: 7 constant columns remain and scale cleanly to 0.0 with 0 NaNs (`scale_ = 1.0`). Non-constant columns: max $|mean| = 1.39 \times 10^{-16}$, min std $= 1.0$, max std $= 1.0$.
  - Pipeline runtime: 0.094s total (load 0.033s, split 0.004s, clean 0.027s, encode 0.016s, scale 0.013s).
- **Surprise:**
  1. Section 3 Items 3 & 4 confirmed: Cleaning removes exclusively suspicious accounts (829 lost from suspicious, 0 from benign), confirming Item 3. The class count shift between raw (7,662/2,179) and post-cleaning (7,662/1,350) is driven entirely by suspicious account loss, confirming Item 4.
  2. Unexplained +22 suspicious gap: The paper reports 1,328 suspicious accounts after cleaning (851 dropped). The CSV has 1,350 suspicious accounts remaining after dropping all numeric NaNs, leaving an unexplained difference of 22 suspicious accounts.
  3. Paper claim of duplicate removal NOT confirmed: The paper claims duplicate addresses were removed in Section 4.1. However, 25 duplicate address pairs (50 rows, all benign) remain in the CSV; removing them would drop benign rows to 7,637, contradicting the paper's post-cleaning count of 7,662 benign accounts.

## 2026-10-04: Phase 1.3 Confusion matrix audit and evaluation harness verification

- **What:** Executed `python -m src.evaluate` analyzing the six paper confusion matrices from Tables 8-9 and Figures 9-14; output saved to `results/phase1_3_paper_confusion_check_20261003_213041.json`.
- **Why:** Investigate Section 3 item 5 (apparent test-set class total discrepancy across models) and verify formula replication for accuracy, MAE, precision, recall, and F1.
- **Number:**
  - Total test samples: $N = 3,065$ across all six models.
  - Printed row sums invariant: $TP + FP = 1,544$ and $TN + FN = 1,521$ across all six confusion matrices without exception.
  - Formula precision: Equations 12-15 applied to the printed cells reproduce reported Accuracy, Precision, Recall, and F1 within $0.00086$ ($\le 0.002$ tolerance) across all six models.
  - MAE alignment: Recomputed MAE matches reported values within $0.006$ (discrepancy explained entirely by the paper rounding MAE to 2 decimal places, e.g. $0.0248 \rightarrow 0.03$, $0.2551 \rightarrow 0.26$, $0.0078 \rightarrow 0.01$).
  - Section 3 item 5 resolution: Under the standard reading (where $TP + FN$ represents actual positive accounts), positive totals appeared to fluctuate wildly between 1,528 and 1,788. Under the printed row reading, $TP + FP = 1,544$ and $TN + FN = 1,521$ are perfectly constant across all models.
- **Surprise:**
  1. Section 3 item 5 confirmed as explained by labelling, not an inconsistent test set: The test set was fixed ($1,544$ benign and $1,521$ suspicious, matching Section 5 text). The apparent contradiction arose because the authors interpreted scikit-learn's standard confusion matrix layout `[[TN, FP], [FN, TP]]` as `[[TP, FP], [FN, TN]]` (inferred from the cell identities and sums).
  2. Reported "Precision" and "Recall" in Tables 8-9 mathematically track the benign class (1,544) under standard definitions, though presented as general detection performance.
