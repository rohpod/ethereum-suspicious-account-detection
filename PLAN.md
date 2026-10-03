# Ethereum Suspicious-Account Detection (PSO + GA + XGBoost)

Context file for coding agents. Read fully before any task. Follow it literally.

## 1. Project

Reproduce, benchmark, improve and extend the framework in **El-Attar et al., "An Optimized Framework for Detecting Suspicious Accounts in the Ethereum Blockchain Network", Cryptography 2025, 9, 63** (`docs/paper.pdf`).

Goals: working open-source repo, honest benchmark against the paper, quantified improvements, short write-up, plus an on-chain smart-contract layer (Phase 4). Academic mini project showcase. 

## 2. Paper facts (source of truth; section numbers refer to the paper)

**Data (4.1, Data Availability Statement):** Kaggle dataset, URL given in the paper's Data Availability Statement (`https://www.kaggle.com/code/chiticariucristian/fraud-detection-ethereum-transactions/input`; verify it resolves). 9,841 accounts, 7,663 benign, 2,178 suspicious.

**Pipeline as published (Algorithm 6, 4.2):** clean (drop missing) -> z-score -> PSO feature selection on the whole dataset -> SMOTE -> 80/20 split -> per model: GA tuning -> train -> evaluate. Models: XGBoost, SVM, Isolation Forest (IF).

**Numbers stated:**
- Cleaning: 851 missing values across 25 features; dropped -> 8,990 accounts (7,662 benign, 1,328 suspicious per 4.2.4).
- SMOTE -> 15,324 accounts (7,662 + 7,662). Split 80/20: train 12,259, test 3,065 (1,544 benign / 1,521 suspicious per Section 5).
- PSO (4.2.3): 14 of 49 features, fitness RMSE 0.3443 after 50 iterations; inertia w(t) = alpha * w(t-1), alpha = 0.99; velocity Eq. 10, position Eq. 11, velocity clamp Vmax = (a_max - a_min)/SEG.
- PSO selected features (Table 2): Avg Min Between Received Transactions; Time Difference Between First and Last (Mins); Sent Transaction; Received Transaction; Unique Received From Addresses; Unique Sent To Addresses; Avg Value Sent; Total Transactions (including contract creation); Total ERC20 Transactions; Total ERC20 Sent (in Ether); Unique ERC20 Sent Address; Minimum Value Sent (ERC20); Maximum Value Sent (ERC20); Average Value Sent (ERC20).
- GA (4.3, Algorithm 2, Tables 3-4): population 50, 20 generations, roulette-wheel selection, crossover, mutation. Search space: XGBoost (learning rate, max_depth, n_estimators, subsample ratio in text; Table 1/5 list n_estimators, gamma, min_child_weight, colsample_bytree, max_depth, reg_lambda, learning rate); SVM (C, gamma; text also mentions kernel type); IF (contamination, max_samples, n_estimators).
- Metrics reported: accuracy, MAE, precision, recall, F1, AUC-ROC, confusion matrix.

**Reference results to compare against (do not force-match):**

| Model | Stage | Acc | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| XGBoost | default | 0.975 | 0.97 | 0.98 | 0.975 | 0.97 |
| SVM | default | 0.744 | 0.826 | 0.713 | 0.765 | 0.71 |
| IF | default | 0.694 | 0.732 | 0.683 | 0.707 | 0.53 |
| XGBoost | GA | 0.992 | 0.992 | 0.993 | 0.992 | 0.99 |
| SVM | GA | 0.87 | 0.869 | 0.872 | 0.87 | 0.86 |
| IF | GA | 0.824 | 0.824 | 0.827 | 0.825 | 0.79 |

Hyperparameters, default -> GA (Tables 5-7):
- XGBoost: n_estimators 100->79, gamma 0->0.28, min_child_weight 1->1.94, colsample_bytree 1->0.58, max_depth 6->5, reg_lambda 1->0.77, learning_rate 0.3->0.55.
- SVM: C 0.1->10.92, gamma 0.1->13.262.
- IF: contamination 0.1->0.3, max_samples 256 ("auto")->1000, n_estimators 100->89.

Table 10 baselines (re-run in the paper on the same dataset): LOF 0.949, CART 0.810, default XGBoost 0.975 accuracy.

## 3. Paper inconsistencies (handle exactly as stated; never silently "fix")

1. **SMOTE contradiction.** Section 5 says SMOTE was applied only to training folds, but 15,324 total, 12,259/3,065 split and a balanced test set (1,544/1,521) imply SMOTE before the split, so the test set contains synthetic accounts. Pipeline A follows the numbers.
2. **Leakage order.** Scaling, PSO ("on the whole dataset", Algorithm 6 step 2) and SMOTE all precede the split.
3. **Cleaning removes mostly suspicious accounts.** 2,178 -> 1,328 suspicious (850 lost) vs 7,663 -> 7,662 benign (1 lost). Missingness is strongly label-linked.
4. **Class counts differ** between 4.1 (7,663/2,178) and 4.2.4 (7,662/1,328); the latter is post-cleaning.
5. **Confusion matrices disagree on test-set class totals.** Suspicious (TP+FN) / benign (FP+TN): XGBoost default 1,528/1,537; XGBoost GA 1,542/1,523; SVM default 1,788/1,277; SVM GA 1,537/1,528; IF default 1,654/1,411; IF GA 1,540/1,525; Section 5 says 1,521/1,544. A single fixed test set cannot do this.
6. **Figures 13/14 text is swapped** (IF before/after GA descriptions).
7. **Fitness metrics unclear.** PSO fitness is RMSE (0.3443, about 88% accuracy) while Algorithm 1 maximises fitness; GA fitness peaks at 0.918 (Table 4) yet XGBoost reports 0.992. Table 4 also has obvious typos (generation 6 min 0.982; generation 13 std 0.1524).
8. **Feature count.** Text says 49; Table A2 lists 45 columns besides Index, Address, Flag (verify). Index and Address must never be model inputs. Two columns (Most Sent/Received Token Type) are categorical; encoding is unspecified.
9. **IF is unsupervised** but tuned with a label-based fitness and judged like a classifier.
10. **GA budget.** Text says 50 x 20 = 1,000 evaluations per classifier; Table 4 shows about 600 evaluations in total.
11. **Table 10** baselines equal the paper's own default XGBoost (0.975); "standard deviation of accuracy" 0.109 is unexplained. Cross-validation, robustness, scalability and runtime claims have no reported numbers.
12. **"50 million transactions"** vs 9,841 account-level rows: confirm what the features aggregate.
13. **Unspecified (log each in `docs/ASSUMPTIONS.md`):** PSO swarm size, c1, c2, SEG; how continuous PSO positions become a feature subset; which model computes PSO fitness; GA crossover/mutation rates, number of parents, fitness metric and CV folds; SMOTE settings; SVM kernel; XGBoost subsample; categorical encoding; random seeds; library versions.

## 4. Hard constraints

- Everything free. No paid APIs, services or compute.
- Dev machine: MacBook Air M2, 8 GB RAM. Heavy PSO/GA runs go to Colab/Kaggle free tier; code must run in both.
- ML: Python scripts plus one config file. Notebooks only for EDA (`notebooks/`).
- Smart contracts: Solidity + Hardhat (JavaScript/TypeScript), local node first, then Sepolia with faucet ETH.
- Keep changes small and reviewable.

## 5. Anti-hallucination rules (mandatory)

1. **Paper is the source of truth.** Read the matching paper section and appendix tables before implementing a stage. Cite the section in the docstring.
2. **Never invent a value.** For anything unspecified (Section 3, item 13), add to `docs/ASSUMPTIONS.md`: parameter, chosen value, reason, "not specified in paper". Ask the user if it materially changes results.
3. **Never fabricate results.** Every number in docs, README or write-up must come from a file in `results/`. If a run has not happened, write "not yet run".
4. **Never fabricate** URLs, dataset links, package names, versions, contract addresses or API behaviour. If the dataset cannot be found at the paper's URL, stop and ask; do not substitute another dataset without logging it.
5. **Verify before claiming.** Run code and tests, show real output, then say done.
6. **Scope discipline.** Change only what the task asks. Report unrelated bugs; do not fix them unasked.
7. **No new dependency** without adding it to `requirements.txt` / `package.json` and noting why in the commit message.
8. **Do not "correct" the paper silently.** Follow the paper as written in Pipeline A, record the inconsistency, and test it in Pipeline B.
9. **When uncertain, ask.** One short question beats a wrong guess.

## 6. Security rules

- Never commit private keys, mnemonics, API keys or `.env`. `.env` stays in `.gitignore`; commit `.env.example` with empty values.
- Throwaway testnet wallet only. Never mainnet.
- Do not commit the raw dataset if its licence forbids it. `data/` stays gitignored.

## 7. Repo layout

```
config/config.yaml        # seeds, paths, PSO/GA budgets, CV settings
data/                     # gitignored
src/
  data.py                 # load, clean, encode
  preprocess.py           # scale, SMOTE, split
  pso.py                  # feature selection
  ga.py                   # hyperparameter tuning (DEAP-style logbook)
  models.py               # XGBoost, SVM, IsolationForest, LOF, CART wrappers
  evaluate.py             # metrics, CV, consistency checks, result writing
  pipeline_a.py           # paper-literal (follows paper's numbers)
  pipeline_b.py           # leakage-safe
  bridge.py               # Phase 4: score addresses, push on-chain
notebooks/                # EDA only
results/                  # one CSV/JSON per run, never hand-edited
contracts/                # Hardhat project
docs/                     # paper.pdf, FINDINGS.md, QUESTIONS.md, ASSUMPTIONS.md
tests/                    # pytest
README.md
PLAN.md
```

## 8. Global convention

- **Config-driven:** no magic numbers in code.
- **Seeds:** one global seed from config applied to numpy, sklearn, xgboost, imbalanced-learn, PSO, GA.
- **Pipeline A (replication):** order and numbers exactly as the paper's numbers imply: clean, z-score, PSO on all data, SMOTE on all data, 80/20 split, GA, evaluate on the (balanced, synthetic-containing) test set.
- **Pipeline B (leakage-safe):** split first (stratified, test set keeps natural class ratio, no SMOTE on it). Fit scaler, SMOTE, PSO and GA only on training data (inner stratified CV for PSO/GA fitness). Test set used once per final evaluation.
- **Metrics (fixed, never change):** accuracy, MAE, precision, recall, F1, AUC-ROC (as in the paper) plus PR-AUC and MCC, and the confusion matrix. On Pipeline B always report the majority-class baseline and per-class recall of the suspicious class.
- **Consistency checks (run on every evaluation):** TP+FN and FP+TN must be identical across all models on the same test set; accuracy recomputed from the confusion matrix must match the reported value.
- **Logging:** each run writes `results/<phase>_<name>_<timestamp>.json|csv` with config snapshot, metrics and wall-clock time per stage.
- **Caching:** save PSO-selected features and GA best parameters; reuse, never recompute silently.
- **GA logbook:** record per generation: gen, nevals, avg, std, min, max (same columns as the paper's Table 4).
- **FINDINGS.md:** append an entry after every experiment: what, why, number, surprise.
- **Git:** small commits, `phaseX.Y: what changed`. Tag at each phase end: `v1-baseline`, `v2-optimised`, `v3-limitations`, `v4-onchain`, `v5-final`.

## 9. Task workflow (every chunk)

1. Read this file, the relevant paper section, `docs/ASSUMPTIONS.md`.
2. Investigate read-only; state what exists and what is missing.
3. Propose a short plan (files, tests). Wait for approval if non-trivial.
4. Implement only that scope.
5. Run tests and the relevant script; capture real output.
6. Update `results/`, `docs/FINDINGS.md`, `docs/ASSUMPTIONS.md`.
7. Summarise: changes, observed results, flagged out-of-scope issues, open questions.
8. Commit. Tag only at phase end.

## 10. Phases

### Phase 1: As-is benchmark
Goal: reproduce the paper as written (Pipeline A only) and set the baseline. No improvements. No leakage-safe pipeline in this phase (that is Phase 2.0).

- **1.1 Setup and data.** Skeleton, config, requirements, seeds. Download the dataset from the paper's URL. Assert 9,841 rows, 7,663 benign, 2,178 suspicious; count feature columns (paper: 49; Table A2: 45 plus Index/Address/Flag). Exclude Index, Address, Flag from inputs. Encode the two categorical token-type columns; log the method as an assumption. Report mismatches; do not adjust data to fit.
- **1.2 Cleaning and scaling.** Drop rows with missing values (expect 851 rows, 8,990 left). Log class counts before and after (expect suspicious 2,178 -> 1,328). Remove duplicate/contradictory rows if any. Z-score.
- **1.3 Models and evaluation harness.** `models.py`: XGBoost (Table 5 defaults), SVM (C 0.1, gamma 0.1), IF (contamination 0.1, max_samples 256, n_estimators 100). `evaluate.py`: the fixed metrics and the consistency checks from Section 8. Test on small synthetic data; real default results are recorded in 1.5, after PSO, SMOTE and the split.
- **1.4 PSO.** Implement Algorithm 1 and Eq. 2-11, 50 iterations, alpha 0.99. Fitness as RMSE (as in the paper); log how positions map to a feature subset. Run on the whole cleaned dataset (Pipeline A). `pso.py` must take X, y and config as arguments (no data loading inside) so Phase 2.0 can reuse it on training data only. Compare the subset with Table 2 (overlap, not equality); target size about 14.
- **1.5 SMOTE, split and default baselines (Pipeline A).** SMOTE on all data -> 15,324; 80/20 split -> 12,259 / 3,065. Check test counts near 1,544 / 1,521. Then record the default-model results (XGBoost, SVM, IF) on this test set using the PSO-selected features.
- **1.6 GA.** Population 50, 20 generations, roulette selection, crossover, mutation (rates assumed and logged). Search spaces from Section 2. `ga.py` takes data as arguments, like `pso.py`. Log the GA logbook; compare best parameters with Tables 5-7.
- **1.7 Comparison baselines.** LOF and CART (and default XGBoost) as in Table 10.
- **1.8 Benchmark table and checks.** Paper vs Pipeline A, all three models, before and after GA. Run consistency checks. Record everything not reproduced and why.

Done when: table in `results/` (paper vs A), `FINDINGS.md` marks each Section 3 inconsistency as confirmed, not confirmed, or needs Pipeline B (Phase 2), tag `v1-baseline`.

### Phase 2: Leakage-safe pipeline, speed and accuracy
Goal: build the leakage-safe Pipeline B, then make the pipeline faster and better, measured against the Pipeline B baseline.

- **2.0 Leakage-safe Pipeline B.** Implement `pipeline_b.py` as defined in Section 8: split first (stratified, natural class ratio), fit scaler, SMOTE, PSO and GA on training data only (inner stratified CV for PSO/GA fitness), test set used once. Reuse `pso.py` and `ga.py` from Phase 1. Run the leakage ladder: A -> SMOTE after split only -> full B. Record the Pipeline B baseline for all three models (default and GA) plus LOF and CART, with the majority-class baseline and per-class recall of the suspicious class. Report the A-vs-B gap and update `FINDINGS.md` with which Section 3 items it confirms.
- **2.1 Profile.** Time each stage (GA = about 600 to 1,000 model fits per classifier; SVM scales poorly with sample count).
- **2.2 Speed.** XGBoost `tree_method="hist"`, early stopping, parallel fitness evaluation, cheaper proxy model in the PSO fitness, caching repeated GA individuals, smaller budgets where metrics hold.
- **2.3 Accuracy.** Class weights instead of SMOTE, threshold tuning on validation folds, optional LightGBM.
- **2.4 Ablation.** Same-budget comparison: PSO+GA vs random search vs Optuna vs no feature selection. State plainly if PSO/GA do not win.
- **2.5 Re-benchmark.** Same protocol on Pipeline B; report speed-up (%) and metric deltas against the 2.0 baseline.

Done when: A-vs-B table (with leakage ladder) and before/after optimisation table with timings, tag `v2-optimised`.

### Phase 3: Limitations
Goal: address chosen weaknesses. Pick at most two or three; record choice and reason in `FINDINGS.md` first.

- **3.1 Missing-data handling (recommended).** Cleaning removed 850 of 2,178 suspicious accounts. Compare dropping vs imputation (+ missing-indicator features); report class counts and metrics on Pipeline B.
- **3.2 Test the paper's untested claims.** Cross-validation stability, noise and missing-feature robustness, runtime scaling. Report numbers with standard deviations.
- **3.3 Explainability.** SHAP on final XGBoost; compare with PSO-selected features.
- **3.4 Temporal robustness.** Time-based split if timestamps allow (paper lists novel patterns as a limitation).
- **3.5 Isolation Forest fairness.** Evaluate as an anomaly scorer (PR-AUC, thresholds), not as a supervised classifier.
- **3.6 Data expansion (optional, costly).** Small fresh sample via free Etherscan API tier; rate limits; key in `.env`; check label quality and overlap.
- **3.7 Graph features (optional).** Network-structure features beyond degree-style counts.

Done when: each chosen item has a result and an honest limitation note, tag `v3-limitations`.

### Phase 4: Smart contracts (on-chain risk layer)
Goal: add "prevention". Inference stays off-chain.

- **4.1 Toolchain.** Hardhat in `contracts/`, Solidity 0.8.x, OpenZeppelin access control. Pin versions via lockfile.
- **4.2 `RiskRegistry.sol`.** Only an authorised oracle writes. Per address: risk score (basis points), flagged bool, model-version hash, timestamp. Batch update, events. No raw features on-chain.
- **4.3 Tests.** Access control, threshold edge cases, batch updates, events, overwrite.
- **4.4 Bridge (`src/bridge.py`).** Load final model, score addresses, push batched transactions via web3.py using `.env`. Local node first.
- **4.5 `RiskGate` / `GuardedVault`.** Modifier reading the registry; delay or reject transfers involving addresses above a threshold. Prefer delay/flag over hard block.
- **4.6 Deploy.** Local, then Sepolia (throwaway wallet). Verify on Etherscan. Record the real address in `FINDINGS.md`.
- **4.7 Gas report.** Single vs batched updates; gate overhead.
- **Document limitations:** off-chain inference, single trusted oracle (mention multisig/model-hash commitment), false-positive impact, label staleness.

Done when: verified testnet deployment, passing tests, demo script (score, push, gated transfer), gas table, tag `v4-onchain`.

### Phase 5: Final optimisation and closing gaps
- **5.1 Freeze benchmark.** Final table: paper vs A vs B vs optimised vs extensions; all numbers from `results/`.
- **5.2 Clean-clone rerun.** Fresh clone and environment; follow README only; fix every gap.
- **5.3 Write-up.** Introduction, Related Work, Method, Experiments, Replication Findings (Section 3 items, stated neutrally as "we could not reconcile ..." with paper section references), On-chain Layer, Limitations, SDG 16 mapping (target 16.4), References (base paper, component repos).
- **5.4 README.** Headline results table, setup, reproduction steps, repo map, limitations, licence.
- **5.5 Release.** Tag `v5-final`; confirm no secrets or restricted data committed.

## 11. Reference repos (components only)

eltontay/Ethereum-Fraud-Detection, NAKhoa819/ADCC-Bench, sepandhaghighi/Ethereum-Fraud-Detection-Visualization, lindan113/EthereumHeist, lindan113/xblock-network_analysis. Read for ideas; do not copy code without checking the licence and citing it.
