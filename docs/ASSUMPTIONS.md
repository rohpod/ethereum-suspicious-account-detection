# Assumptions

Every value the paper does not specify, or any deviation from it, is logged here.

| Parameter | Value | Reason | Source |
|---|---|---|---|
| seed | 42 | not specified in paper | NA |
| Python version | 3.14 | local dev environment | NA |
| Dataset License | DbCL v1.0 | NA | Kaggle |
| label polarity | FLAG 1 = suspicious, 0 = benign | Standard fraud classification convention; matches suspicious counts in paper (~2,178 suspicious) | not specified in paper |
| column-name normalisation | strip, lowercase, non-alphanumerics to '_', collapse repeats | Raw CSV contains leading/trailing spaces, typos, and unclosed parentheses | not specified in paper |
| id columns excluded | ['Unnamed: 0', 'Index', 'Address'] excluded from X | Account identifiers must not leak into model features; Table A2 lists Index/Address separately | not specified in paper |
| candidate features kept | All 47 candidate features kept (45 numeric + 2 categorical) | Raw CSV contains 47 features excluding ID and label; Table A2 lists 45, text cites 49 | not specified in paper |
| token encoding method | TokenFrequencyEncoder (frequency = count / n_fit_rows) | Two categorical token features have unspecified encoding in paper; frequency encoding yields numeric features without high dimensionality | not specified in paper |
| token category handling | NaN -> '__missing__', empty/whitespace -> '__blank__', '0' kept as distinct category | Token columns have 2,697 / 871 NaNs, 1,191 / 21 whitespace strings, and 4,399 '0' strings | not specified in paper |
| unseen token category handling | 0.0 | Unseen token types in test/evaluation distribution have 0 frequency in training distribution | not specified in paper |
| validation policy | Report mismatches, do not adjust data | Anti-hallucination rule 8: preserve raw data characteristics and log discrepancies rather than forcing paper numbers | not specified in paper |
| dataset filename and source | transaction_dataset.csv from Kaggle | Data Availability Statement URL resolves to Kaggle dataset with this exact filename | not specified in paper |
| cleaning policy | Drop rows with NaN in numeric feature columns only; retain token NaNs | Dropping all NaNs would drop 1,891 benign rows leaving only 5,771; numeric drop matches 7,662 benign exactly | Section 4.2.1 |
| duplicates retention | Duplicate feature vectors and duplicate addresses NOT removed | Removing duplicate addresses reduces benign rows to 7,637; removing duplicate features drops 23-30 benign and 243 suspicious; paper's 7,662 benign requires retention | Section 4.1 |
| z-score feature scope | Applied to all 47 features including the 2 encoded token columns | Paper specifies z-score normalization across all features before PSO | Section 4.2.2 |
| constant columns retention | 7 zero-variance columns retained in feature matrix | StandardScaler safely scales zero-variance features (scale_=1) to 0.0 without NaNs; paper does not report dropping constant columns before PSO | Section 4.2.2 |
| scaler fitting scope (Pipeline A) | Fitted on all 9,012 cleaned rows before splitting | Follows paper Algorithm 6 and Section 4.2 order where standardization precedes PSO, SMOTE, and data splitting | Section 4.2.2, Algorithm 6 |
