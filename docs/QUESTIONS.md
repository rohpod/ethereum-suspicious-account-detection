# Open Questions

## Phase 1.2: Cleaning Gap (829 vs. 851 Suspicious Rows)

- **Context:**
  - Paper Section 4.2.1 states: *"This study identified 851 instances of missing data across 25 features. To address this, missing data was dropped, resulting in a dataset of 8990 accounts."*
  - Paper Section 4.2.4 reports the post-cleaning class balance as **7,662 benign** and **1,328 suspicious** accounts (7,662 + 1,328 = 8,990).
  - Comparing the raw counts (7,662 benign, 2,179 suspicious in CSV; 7,663 benign, 2,178 suspicious in paper Section 4.1) against post-cleaning indicates that 850 (or 851) suspicious accounts and 0 (or 1) benign accounts were dropped.
- **Observed Data:**
  - In `data/transaction_dataset.csv`, dropping rows with missing values in the 23 numeric ERC20 features drops exactly **829 rows**, all of which have `FLAG = 1` (suspicious). This leaves **1,350 suspicious** accounts and 7,662 benign accounts (total 9,012 accounts).
  - Dropping rows with missing values across all 25 ERC20 features (including the 2 categorical token type columns) drops 2,720 rows (leaving 7,121 accounts: 5,771 benign and 1,350 suspicious).
- **Question for Phase 1.2:**
  - What accounts for the **22-row gap** between the 829 rows missing numeric ERC20 features and the paper's 851 dropped rows (1,350 - 1,328 = 22 suspicious accounts)?
  - Did the original authors apply an additional filtering step (such as dropping specific outliers, deduplicating certain suspicious accounts, or another missing-value heuristic) to drop those 22 suspicious accounts?
  - For Phase 1.2, should Pipeline A attempt to identify and isolate those 22 accounts, or should it proceed with dropping the 829 numeric-missing rows (documenting the 22-row difference)?
