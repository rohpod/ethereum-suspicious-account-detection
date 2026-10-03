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
