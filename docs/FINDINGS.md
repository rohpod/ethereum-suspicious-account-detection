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
  1. Section 3 item 5 confirmed as explained by labelling, not an inconsistent test set: The test set was fixed ($1,544$ benign and $1,521$ suspicious, matching Section 5 text). The apparent contradiction arose because the authors interpreted scikit-learn's standard confusion matrix layout `[[TN, FP], [FN, TN]]` as `[[TP, FP], [FN, TN]]` (inferred from the cell identities and sums).
  2. Reported "Precision" and "Recall" in Tables 8-9 mathematically track the benign class (1,544) under standard definitions, though presented as general detection performance.

## 2026-10-04: Phase 1.4 Particle Swarm Optimization (Pipeline A feature selection)

- **What:** Executed `python -m src.pso` on the full cleaned and scaled dataset (9,012 accounts, 47 features); output saved to `results/phase1_4_pso_20261003_220124.json` and cached to `results/cache/pso_pipeline_a.json`.
- **Why:** Implement and evaluate PSO feature selection per paper Section 4.2.3, Equations 2-11, Algorithm 1, Figure 4, and Table 2 under Pipeline A semantics.
- **Number:**
  - Selected features: 22 features selected vs paper's 14 features ($\Delta = +8$).
  - Best RMSE: 0.0976 after 50 iterations vs paper reported 0.3443 (substantially lower error; RMSE 0.0976 corresponds to ~99.0% CV accuracy on hard labels, whereas RMSE 0.3443 implies ~88.1% accuracy).
  - Majority baseline RMSE: 0.3870 (always-predicting-benign baseline on 1,350 suspicious / 9,012 total accounts, $\sqrt{1350/9012}$).
  - All 47 features RMSE: 0.1214 (DecisionTreeClassifier 3-fold CV with all features).
  - Table 2 features RMSE:
    - Candidate 1 (with `erc20_uniq_sent_addr`): 0.1914.
    - Candidate 2 (with `erc20_uniq_sent_addr_1`): 0.1920.
    - 13 unambiguous features: 0.1932.
  - Table 2 overlap:
    - Unambiguous features overlap: 6 of 13 matched (`time_diff_between_first_and_last_mins`, `unique_received_from_addresses`, `unique_sent_to_addresses`, `avg_val_sent`, `total_erc20_tnxs`, `erc20_total_ether_sent`), Jaccard similarity = 0.2069.
    - Ambiguous candidate resolution: For `Unique ERC20 Sent address`, Candidate 1 (`erc20_uniq_sent_addr`) was selected by PSO, while Candidate 2 (`erc20_uniq_sent_addr_1`) was not. Including Candidate 1 brings total Table 2 overlap to 7 of 14 features (50.0%).
    - Missing from Table 2: 7 features (`avg_min_between_received_tnx`, `sent_tnx`, `received_tnx`, `total_transactions_including_tnx_to_create_contract`, `erc20_min_val_sent`, `erc20_max_val_sent`, `erc20_avg_val_sent`).
    - Extra features: 16 features selected by PSO not in Table 2 (including `erc20_most_sent_token_type`, `total_ether_sent`, `total_ether_received`, `min_val_sent`, etc.).
  - Fitness evaluations: 1,530 total (30 initial swarm + 50 iterations $\times$ 30 particles).
  - Wall-clock runtime: 39.63s for PSO optimization (39.85s total pipeline).
- **Surprise:**
  1. Selected subset does not match Table 2 exactly: PSO selected 22 features with an overlap of 6/13 unambiguous (7/14 including Candidate 1). This confirms the expectation in PLAN.md Section 10 Phase 1.4 ("Compare the subset with Table 2 (overlap, not equality); target size about 14").
  2. Optimization significantly outperforms paper reported fitness: Best RMSE achieved is 0.0976 vs paper's 0.3443. Evaluating the paper's Table 2 features directly yields RMSE 0.1914, which is also lower than 0.3443. This indicates the paper either used a weaker classifier for PSO fitness evaluation (e.g. shallow tree, linear model, or different CV split) or terminated early.
  3. Algorithm 1 oddities confirmed (Section 3 Item 7):
     - Line 5 uses `>` ("if fitness(x) > fitness(x*)") despite the text stating "search for the minimum objective function value" and reporting RMSE minimization.
     - Line 7 reassigns `x**_i = x*_i` locally per particle rather than updating a global swarm best $g^*$.
     - Line 9 loops `d = 1 to iteration` rather than over feature dimensions $d = 1 \dots D$.
     - No discretization threshold (e.g. $> 0.5$) or empty-subset penalty is specified in the text or algorithm.
  4. Caching: Deterministic caching verified (`results/cache/pso_pipeline_a.json`), reducing re-runs from ~40s to ~0.001s with hash validation.

## 2026-10-04: Phase 1.5 SMOTE, split, and default baselines (Pipeline A)

- **What:** Executed `python -m src.pipeline_a` with config in `config/config.yaml`; output saved to `results/phase1_5_pipeline_a_defaults_20261004_031101.json`.
- **Why:** Implement SMOTE balancing and train/test split in `src/preprocess.py`, then benchmark default XGBoost, SVM, and Isolation Forest models on both PSO-selected features (22 features) and paper Table 2 reference features (14 features) per paper Sections 4.2.4, 4.2.5, and Table 8 under Pipeline A order.
- **Number:**
  - SMOTE balance: Raw cleaned data (9,012 rows: 7,662 benign, 1,350 suspicious) balanced to exactly 15,324 rows (7,662 benign, 7,662 suspicious), adding exactly 6,312 synthetic suspicious rows. Matches paper Section 4.2.4 figure (15,324 total, 7,662 each) with $\Delta = 0$.
  - Train/test split: Unstratified 80/20 split (`seed=42`, `shuffle=True`, `stratify=None`) yields exactly 12,259 train and 3,065 test rows, matching paper Section 4.2.5 figures (12,259 / 3,065) with $\Delta = 0$.
  - Test class balance: 1,530 benign (0) and 1,535 suspicious (1) vs paper's reported 1,544 benign and 1,521 suspicious ($\Delta = -14$ benign, $+14$ suspicious).
  - Test set synthetic leakage (PLAN.md Section 3 Item 1): Exactly 1,245 of the 3,065 test accounts (40.6%) are synthetic rows generated by SMOTE prior to the split. Among the 1,535 suspicious test accounts, 1,245 (81.1%) are synthetic samples; only 290 real suspicious accounts remain in the test set.
  - Baseline Model Performance on PSO feature set (22 features) vs paper Table 8 defaults:
    - XGBoost: Accuracy = 0.9967 (paper: 0.975, $\Delta = +0.0217$), Precision = 0.9993 (paper: 0.97, $\Delta = +0.0293$), Recall = 0.9941 (paper: 0.98, $\Delta = +0.0141$), F1 = 0.9967 (paper: 0.975, $\Delta = +0.0217$), AUC = 0.9999 (paper: 0.97, $\Delta = +0.0299$), MAE = 0.0033. Confusion matrix: `[[1529, 1], [9, 1526]]`.
    - SVM: Accuracy = 0.9377 (paper: 0.744, $\Delta = +0.1937$), Precision = 0.9275 (paper: 0.826, $\Delta = +0.1015$), Recall = 0.9498 (paper: 0.713, $\Delta = +0.2368$), F1 = 0.9385 (paper: 0.765, $\Delta = +0.1735$), AUC = 0.9698 (paper: 0.71, $\Delta = +0.2598$), MAE = 0.0623. Confusion matrix: `[[1416, 114], [77, 1458]]`.
    - Isolation Forest: Accuracy = 0.4395 (paper: 0.694, $\Delta = -0.2545$), Precision = 0.2269 (paper: 0.732, $\Delta = -0.5051$), Recall = 0.0495 (paper: 0.683, $\Delta = -0.6335$), F1 = 0.0813 (paper: 0.707, $\Delta = -0.6257$), AUC = 0.1869 (paper: 0.53, $\Delta = -0.3431$), MAE = 0.5605. Confusion matrix: `[[1271, 259], [1459, 76]]`.
  - Baseline Model Performance on Table 2 reference feature set (14 features) vs paper Table 8 defaults:
    - XGBoost: Accuracy = 0.9896 (paper: 0.975, $\Delta = +0.0146$), Precision = 0.9883 (paper: 0.97, $\Delta = +0.0183$), Recall = 0.9909 (paper: 0.98, $\Delta = +0.0109$), F1 = 0.9896 (paper: 0.975, $\Delta = +0.0146$), AUC = 0.9992 (paper: 0.97, $\Delta = +0.0292$), MAE = 0.0104. Confusion matrix: `[[1512, 18], [14, 1521]]`.
    - SVM: Accuracy = 0.7005 (paper: 0.744, $\Delta = -0.0435$), Precision = 0.6409 (paper: 0.826, $\Delta = -0.1851$), Recall = 0.9140 (paper: 0.713, $\Delta = +0.2010$), F1 = 0.7535 (paper: 0.765, $\Delta = -0.0115$), AUC = 0.8208 (paper: 0.71, $\Delta = +0.1108$), MAE = 0.2995. Confusion matrix: `[[744, 786], [132, 1403]]`.
    - Isolation Forest: Accuracy = 0.4509 (paper: 0.694, $\Delta = -0.2431$), Precision = 0.2658 (paper: 0.732, $\Delta = -0.4662$), Recall = 0.0547 (paper: 0.683, $\Delta = -0.6283$), F1 = 0.0908 (paper: 0.707, $\Delta = -0.6162$), AUC = 0.3223 (paper: 0.53, $\Delta = -0.2077$), MAE = 0.5491. Confusion matrix: `[[1298, 232], [1451, 84]]`.
  - Consistency checks: All passed (`class_totals_consistent_across_models` pos=1535, neg=1530; `accuracy_recomputed_matches_reported`).
  - Total pipeline runtime: 5.20s across both feature sets.
- **Surprise:**
  1. The paper's counts (15,324 / 12,259 / 3,065) are exactly reproducible with SMOTE applied before the split, which is consistent with Section 3 item 1. Whether the authors did this is an inference, since Section 5 says SMOTE was applied only to training folds.
  2. SVM results are very sensitive to the feature set (accuracy 0.9377 on our 22 PSO features, 0.7005 on Table 2's 14). Neither feature set reproduces the paper's SVM default (accuracy 0.744) with a consistent precision/recall pattern, so we cannot say which feature set the paper used.
  3. Default Isolation Forest failure explained by contamination parameter: With `contamination=0.1`, Isolation Forest flags only ~10% of samples as anomalous (predicting ~335 anomalies out of 3,065 test points). But the test set is 50.1% suspicious (1,535/3,065). Predicting only ~10% positive guarantees recall $\le 0.055$ and accuracy $\approx 0.44-0.45$. In paper Table 7, GA tuning adjusts contamination from 0.1 to 0.3, partially mitigating this under-prediction.
  4. Test set class split near-exact match: Unstratified 80/20 random split (`seed=42`) produces 1,530 benign and 1,535 suspicious, matching the paper's reported 1,544 benign and 1,521 suspicious within 14 samples without requiring artificial split manipulation.
