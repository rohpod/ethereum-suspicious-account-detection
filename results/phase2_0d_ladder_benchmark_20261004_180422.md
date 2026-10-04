# Master Leakage Ladder Benchmark (Phase 2.0d)

Generated at: 20261004_180422 (UTC).
Consolidates results across published literature, Pipeline A replication, and leakage-safe Pipeline B ladder rungs.

## Source Files

- **phase1_8_benchmark**: `phase1_8_benchmark_20261004_044733.json`
- **phase2_0b_ladder**: `phase2_0b_pipeline_b_ladder_20261004_161846.json`
- **phase2_0c_duplicates**: `phase2_0c_duplicates_20261004_163639.json`
- **phase2_0c_ga_b_isolation_forest**: `phase2_0c_ga_b_isolation_forest_20261004_165411.json`
- **phase2_0c_ga_b_svm**: `phase2_0c_ga_b_svm_20261004_171516.json`
- **phase2_0c_ga_b_xgboost**: `phase2_0c_ga_b_xgboost_20261004_164208.json`
- **phase2_0c_ga_l1_xgboost**: `phase2_0c_ga_l1_xgboost_20261004_164511.json`

## 1. Per-Rung Evaluation Distribution Diagnostics

| Evaluation Rung | Test Set Size | Benign (Class 0) | Suspicious (Class 1) | Synthetic Fraction | Majority Baseline Accuracy | Suspicious Recall | Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pipeline A (Replication)** | 3,065 | 1,530 | 1,535 | 40.62% | 0.4992 | 0.0000 | 50.08% |
| **Rung L1 (SMOTE After Split Only)** | 1,803 | 1,533 | 270 | 0.00% | 0.8502 | 0.0000 | 14.98% |
| **Rung B (Full Leakage-Safe)** | 1,803 | 1,533 | 270 | 0.00% | 0.8502 | 0.0000 | 14.98% |

## 2. Master Leakage Ladder Benchmark Table

| Model | Stage | Metric | Paper | Pipeline A (Replication) | Rung L1 (SMOTE After Split) | Rung B (Full Leakage-Safe) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **XGBoost**        | Default      | Accuracy             | 0.9750 | 0.9967 | 0.9945 | 0.9945 |
|                    |              | Precision            | 0.9700 | 0.9993 | 0.9745 | 0.9815 |
|                    |              | Recall (Suspicious)  | 0.9800 | 0.9941 | 0.9889 | 0.9815 |
|                    |              | F1-Score             | 0.9750 | 0.9967 | 0.9816 | 0.9815 |
|                    |              | ROC-AUC              | 0.9700 | 0.9999 | 0.9998 | 0.9997 |
|                    |              | PR-AUC               | not run | not run | 0.9988 | 0.9986 |
|                    |              | MCC                  | not run | not run | 0.9784 | 0.9782 |
| **XGBoost**        | GA Tuned     | Accuracy             | 0.9920 | 0.9971 | 0.9928 | 0.9911 |
|                    |              | Precision            | 0.9920 | 1.0000 | 0.9639 | 0.9635 |
|                    |              | Recall (Suspicious)  | 0.9930 | 0.9941 | 0.9889 | 0.9778 |
|                    |              | F1-Score             | 0.9920 | 0.9971 | 0.9762 | 0.9706 |
|                    |              | ROC-AUC              | 0.9900 | 0.9999 | 0.9998 | 0.9997 |
|                    |              | PR-AUC               | not run | not run | 0.9991 | 0.9986 |
|                    |              | MCC                  | not run | not run | 0.9721 | 0.9654 |
| **SVM**            | Default      | Accuracy             | 0.7440 | 0.9377 | 0.9218 | 0.9196 |
|                    |              | Precision            | 0.8260 | 0.9275 | 0.6667 | 0.6607 |
|                    |              | Recall (Suspicious)  | 0.7130 | 0.9498 | 0.9556 | 0.9519 |
|                    |              | F1-Score             | 0.7650 | 0.9385 | 0.7854 | 0.7800 |
|                    |              | ROC-AUC              | 0.7100 | 0.9698 | 0.9530 | 0.9523 |
|                    |              | PR-AUC               | not run | not run | 0.6153 | 0.6112 |
|                    |              | MCC                  | not run | not run | 0.7573 | 0.7510 |
| **SVM**            | GA Tuned     | Accuracy             | 0.8700 | 0.9883 | not run | 0.9795 |
|                    |              | Precision            | 0.8690 | 0.9870 | not run | 0.8896 |
|                    |              | Recall (Suspicious)  | 0.8720 | 0.9896 | not run | 0.9852 |
|                    |              | F1-Score             | 0.8700 | 0.9883 | not run | 0.9350 |
|                    |              | ROC-AUC              | 0.8600 | 0.9980 | not run | 0.9983 |
|                    |              | PR-AUC               | not run | not run | not run | 0.9904 |
|                    |              | MCC                  | not run | not run | not run | 0.9245 |
| **Isolation Forest** | Default      | Accuracy             | 0.6940 | 0.4395 | 0.7349 | 0.7399 |
|                    |              | Precision            | 0.7320 | 0.2269 | 0.0517 | 0.0538 |
|                    |              | Recall (Suspicious)  | 0.6830 | 0.0495 | 0.0444 | 0.0444 |
|                    |              | F1-Score             | 0.7070 | 0.0813 | 0.0478 | 0.0487 |
|                    |              | ROC-AUC              | 0.5300 | 0.1869 | 0.1499 | 0.1356 |
|                    |              | PR-AUC               | not run | not run | 0.0904 | 0.0894 |
|                    |              | MCC                  | not run | not run | -0.1056 | -0.1010 |
| **Isolation Forest** | GA Tuned     | Accuracy             | 0.8240 | 0.4897 | not run | 0.3172 |
|                    |              | Precision            | 0.8240 | 0.1463 | not run | 0.0620 |
|                    |              | Recall (Suspicious)  | 0.8270 | 0.0039 | not run | 0.2519 |
|                    |              | F1-Score             | 0.8250 | 0.0076 | not run | 0.0995 |
|                    |              | ROC-AUC              | 0.7900 | 0.2289 | not run | 0.1957 |
|                    |              | PR-AUC               | not run | not run | not run | 0.0920 |
|                    |              | MCC                  | not run | not run | not run | -0.3066 |
| **CART**           | Default      | Accuracy             | 0.8100 | 0.9935 | 0.9806 | 0.9828 |
|                    |              | Precision            | 0.8130 | 0.9928 | 0.9038 | 0.9193 |
|                    |              | Recall (Suspicious)  | 0.8100 | 0.9941 | 0.9741 | 0.9704 |
|                    |              | F1-Score             | 0.8090 | 0.9935 | 0.9376 | 0.9441 |
|                    |              | ROC-AUC              | 0.8100 | 0.9935 | 0.9779 | 0.9777 |
|                    |              | PR-AUC               | not run | not run | 0.8842 | 0.8965 |
|                    |              | MCC                  | not run | not run | 0.9270 | 0.9345 |
| **CART**           | GA Tuned     | Accuracy             | not run | not run | not run | not run |
|                    |              | Precision            | not run | not run | not run | not run |
|                    |              | Recall (Suspicious)  | not run | not run | not run | not run |
|                    |              | F1-Score             | not run | not run | not run | not run |
|                    |              | ROC-AUC              | not run | not run | not run | not run |
|                    |              | PR-AUC               | not run | not run | not run | not run |
|                    |              | MCC                  | not run | not run | not run | not run |
| **LOF**            | Default      | Accuracy             | 0.9490 | 0.4349 | 0.6916 | 0.7022 |
|                    |              | Precision            | 0.9460 | 0.2909 | 0.1478 | 0.1550 |
|                    |              | Recall (Suspicious)  | 0.9540 | 0.0893 | 0.2222 | 0.2222 |
|                    |              | F1-Score             | 0.9490 | 0.1366 | 0.1775 | 0.1826 |
|                    |              | ROC-AUC              | 0.9400 | 0.4012 | 0.5411 | 0.5501 |
|                    |              | PR-AUC               | not run | not run | 0.1742 | 0.1831 |
|                    |              | MCC                  | not run | not run | -0.0030 | 0.0077 |
| **LOF**            | GA Tuned     | Accuracy             | not run | not run | not run | not run |
|                    |              | Precision            | not run | not run | not run | not run |
|                    |              | Recall (Suspicious)  | not run | not run | not run | not run |
|                    |              | F1-Score             | not run | not run | not run | not run |
|                    |              | ROC-AUC              | not run | not run | not run | not run |
|                    |              | PR-AUC               | not run | not run | not run | not run |
|                    |              | MCC                  | not run | not run | not run | not run |

> [!NOTE]
> **Table Footnote**: Pipeline A's test set is balanced and contains synthetic rows (40.62% synthetic accounts), so accuracy and precision are not comparable across Pipeline A and Rungs L1/B. ROC-AUC and PR-AUC provide the closest methodological comparison, and PR-AUC is fundamentally dependent on test prevalence (50.08% in Pipeline A vs. 14.98% in L1/B).

## 3. Genetic Algorithm Optimization Summary

| Model | Ladder Rung | Evaluations | Wall-Clock Time | GA Metric | Best Inner-CV Fitness | Test F1-Score | Tuning-Estimate vs. Test Delta | Best Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost** | Rung B | 663 | 319.90s | `f1_suspicious` | 0.9710 | 0.9706 | -0.0004 | `n_estimators=300, gamma=0.0000, min_child_weight=5.1844, colsample_bytree=0.7517, max_depth=7, reg_lambda=8.9637, learning_rate=0.8657` |
| **XGBoost** | Rung L1 | 663 | 172.53s | `f1_suspicious` | 0.9951 | 0.9762 | -0.0189 | `n_estimators=103, gamma=0.0000, min_child_weight=1.0000, colsample_bytree=0.6383, max_depth=6, reg_lambda=5.1412, learning_rate=0.8942` |
| **Isolation Forest** | Rung B | 637 | 531.72s | `f1_suspicious` | 0.1314 | 0.0995 | -0.0319 | `contamination=0.4979, max_samples=64, n_estimators=145` |
| **SVM** | Rung B | 629 | 1253.88s | `f1_suspicious` | 0.9243 | 0.9350 | +0.0107 | `C=76.6149, gamma=5.0378` |

## 4. Duplicate & Train/Test Overlap Diagnostic Summary

- **Cleaned 47-Feature Matrix ($N=9,012$)**:
  - Redundant duplicate rows (`keep='first'`): **266 (2.95%)** [Benign: 23, Suspicious: 243]
  - Rows in duplicate clusters (`keep=False`): **292 (3.24%)** [Benign: 44, Suspicious: 248]
  - Contradictory groups (identical features with conflicting labels): **0 groups (0 rows)**
- **Train/Test Leakage Overlap (Test Split $N=1,803$)**:
  - Test rows appearing in train split: **57 (3.16%)** [Benign: 9, Suspicious: 48]
  - Strictly unseen test rows: **1746 (96.84%)** [Benign: 1524, Suspicious: 222]
  - Unseen test set majority baseline accuracy: **0.8729** (87.29%)
- **XGBoost Performance on Unseen Test Rows vs. Full Test Set**:
  - Accuracy: 0.9943 (Full: 0.9945, Delta: -0.0002)
  - Precision: 0.9775 (Full: 0.9815, Delta: -0.0040)
  - Recall (Suspicious): 0.9775 (Full: 0.9815, Delta: -0.0040)
  - F1-Score: 0.9775 (Full: 0.9815, Delta: -0.0040)
  - ROC-AUC: 0.9997 (Full: 0.9997, Delta: -0.0001)
  - PR-AUC: 0.9980 (Full: 0.9986, Delta: -0.0006)
  - MCC: 0.9742 (Full: 0.9782, Delta: -0.0040)