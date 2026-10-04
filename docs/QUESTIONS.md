# Open Questions

## Phase 1.2: Cleaning Gap (829 vs. 851 Suspicious Rows) [STILL OPEN AFTER 1.2]

- **Status:** Still open after Phase 1.2.
- **Context:**
  - Paper Section 4.2.1 states: *"This study identified 851 instances of missing data across 25 features. To address this, missing data was dropped, resulting in a dataset of 8990 accounts."*
  - Paper Section 4.2.4 reports the post-cleaning class balance as **7,662 benign** and **1,328 suspicious** accounts (7,662 + 1,328 = 8,990).
  - Comparing raw counts (7,662 benign, 2,179 suspicious in CSV; 7,663 benign, 2,178 suspicious in paper Section 4.1) against post-cleaning indicates that 850 (or 851) suspicious accounts and 0 (or 1) benign accounts were dropped.
- **Empirical Findings in Phase 1.2:**
  - In `data/transaction_dataset.csv`, dropping rows with missing values in the 23 numeric ERC20 features drops exactly **829 rows**, all of which have `FLAG = 1` (suspicious). This leaves **1,350 suspicious** accounts and **7,662 benign** accounts (total 9,012 accounts).
  - The benign count matches the paper's 7,662 figure **exactly** ($\Delta = 0$).
  - Every one of the remaining 1,350 suspicious accounts has **zero missing values** across all 51 columns.
  - No documented cleaning step (deduplication, contradiction removal, or outlier removal) removes 22 suspicious accounts without either removing benign accounts or removing hundreds of suspicious accounts.
  - Removing duplicate addresses reduces benign accounts to 7,637 (contradicting the paper's 7,662).
  - Dropping duplicate feature vectors removes 23 to 30 benign and 243 suspicious accounts (leaving only 1,107 suspicious accounts).
  - Zero contradictory label rows exist.
- **Resolution for Pipeline A:**
  - Pipeline A implements the documented policy: drop rows with missing values in numeric feature columns.
  - This results in 9,012 cleaned rows (7,662 benign, 1,350 suspicious; 829 dropped).
  - The 22-row gap ($1,350 - 1,328 = 22$) remains documented as an unresolvable paper discrepancy.

## Phase 1.3: Target Class Polarity in Reported Precision and Recall

- **Context:**
  - Tables 8-9 report "Precision" and "Recall" across all models.
  - In `results/phase1_3_paper_confusion_check_20261003_213041.json`, mathematical evaluation confirms:
    $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$
    where $TP + FP = 1,544$ for all six models.
  - In Section 5, the paper states the test set contains **1,544 benign** and **1,521 suspicious** accounts.
  - This indicates $TP + FP$ corresponds to the 1,544 benign accounts.
- **Question:**
  - Do the paper's reported "Precision" and "Recall" metrics actually track detection of the benign class (majority class) rather than the suspicious class?
  - In standard anomaly/fraud detection literature, Precision and Recall evaluate the minority/suspicious class.
  - We cannot confirm author intent, but the mathematical identity $TP + FP = 1,544$ strongly infers this orientation.
  - For Pipeline A benchmark reporting, we will report both the paper-reproduced formula values and explicit per-class (suspicious vs benign) metrics in `src/evaluate.py`.

## Phase 1.4: Ambiguous Table 2 Feature Mapping for "Unique ERC20 Sent address"

- **Status:** Documented and isolated in `config/config.yaml` under `pso.table2_ambiguous`.
- **Context:**
  - Table 2 in paper Section 4.2.3 lists 14 selected features. Feature 11 is printed as:
    - Name: *"Unique ERC20 Sent address"*
    - Description: *"Number of unique addresses that received ERC20 token transactions."*
  - In `data/transaction_dataset.csv`, two distinct columns exist with identical semantic headers:
    1. Raw column `' ERC20 uniq sent addr'` $\rightarrow$ normalised `erc20_uniq_sent_addr` (column index 27 in X).
    2. Raw column `' ERC20 uniq sent addr.1'` $\rightarrow$ normalised `erc20_uniq_sent_addr_1` (column index 29 in X).
  - Both columns are present in the Kaggle dataset source. Table A2 in Appendix A lists *"Unique ERC20 Sent Address"* only once, offering no clarification on the duplicate `".1"` column in the CSV.
- **Resolution for Replication:**
  - Strict anti-hallucination policy: We do not guess or arbitrarily choose between the two candidates.
  - `config/config.yaml` isolates the 13 unambiguous matches under `pso.table2_columns` and maps `Unique ERC20 Sent address` to `["erc20_uniq_sent_addr", "erc20_uniq_sent_addr_1"]` under `pso.table2_ambiguous`.
  - Diagnostics in `results/phase1_4_pso_20261003_220124.json` evaluate Table 2 under both candidates:
    - Candidate 1 (`erc20_uniq_sent_addr`): 14-feature RMSE = 0.1914.
    - Candidate 2 (`erc20_uniq_sent_addr_1`): 14-feature RMSE = 0.1920.
    - 13 unambiguous features alone: RMSE = 0.1932.
  - In our PSO execution (seed 42), the algorithm selected Candidate 1 (`erc20_uniq_sent_addr`) and did not select Candidate 2, yielding 7 of 14 features matching Table 2 when Candidate 1 is considered.

