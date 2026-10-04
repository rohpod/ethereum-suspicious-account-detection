# Assumptions

Every value the paper does not specify, or any deviation from it, is logged here.

| Parameter | Value | Reason | Source |
|---|---|---|---|
| seed | 42 | not specified in paper | NA |
| Python version | 3.14 | local dev environment | NA |
| Library versions | scikit-learn==1.9.1, xgboost==3.4.1, imbalanced-learn==0.14.2, numpy==2.5.3, pandas==3.0.6 | Pinned in requirements.txt; paper does not specify package versions; Colab/Kaggle compatibility is untested | not specified in paper |
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
| SVM kernel | rbf | Table 1 mentions radial basis function (RBF) kernel gamma; kernel type otherwise unspecified in paper text | Table 1, Section 4.3 |
| XGBoost objective | binary:logistic | Standard binary classification loss; specific loss function not named in text | not specified in paper |
| XGBoost subsample | 1.0 | Subsample ratio mentioned in text but default value omitted from Table 5 | not specified in paper |
| IF anomaly orientation | anomaly (-1) -> suspicious (1), inlier (+1) -> benign (0); unsupervised fit on X_train only | Isolation Forest is unsupervised; anomalies represent the positive/suspicious class | Section 4.3, Algorithm 5 |
| IF risk score | -score_samples(X) (higher = more suspicious) | Scikit-learn score_samples produces negative values where lower indicates anomalies; negating aligns scores with suspiciousness for AUC-ROC | not specified in paper |
| SVM risk score | decision_function(X) | Distance to separating hyperplane reflects confidence for class 1 (suspicious) without probability calibration overhead | not specified in paper |
| MAE definition | MAE = 1 - accuracy on hard labels (mean \|y_true - y_pred\|) | Matches (FP + FN) / total on binary labels | Tables 8-9, Equation (12) |
| positive class definition | Class 1 (suspicious) as positive class; per-class metrics also tracked | Standard fraud detection convention; macro and benign metrics also computed | not specified in paper |
| model default hyperparameters | Explicit values from Tables 5-7: XGBoost (n_estimators=100, gamma=0, min_child_weight=1, colsample_bytree=1, max_depth=6, reg_lambda=1, learning_rate=0.3); SVM (C=0.1, gamma=0.1); IF (contamination=0.1, max_samples=256, n_estimators=100) | Paper baseline configurations explicitly pinned rather than relying on library defaults | Tables 5-7 |
| PSO swarm size | 30 | Standard swarm population; unspecified in paper | not specified in paper |
| PSO cognitive coefficient c1 | 2.0 | Standard cognitive acceleration parameter in Eq. 10 | not specified in paper |
| PSO social coefficient c2 | 2.0 | Standard social acceleration parameter in Eq. 10 | not specified in paper |
| PSO initial inertia weight w_initial | 0.9 | High starting inertia for initial global exploration; Section 4.2.3 states "initialized with a large value" | not specified in paper |
| PSO SEG parameter | 10 | SEG parameter in Eq. 7 ("SEG is an assumed number") | not specified in paper |
| PSO search space bounds | [0.0, 1.0] | Continuous particle coordinates mapped to binary inclusion | not specified in paper |
| PSO velocity clamping | Vmax = (1.0 - 0.0) / 10 = 0.1, clamped to [-0.1, 0.1] | Computed directly from position bounds and SEG per Eq. 6-7 | Section 4.2.3, Equations (6)-(7) |
| PSO feature selection threshold | position > 0.5 selects feature | Standard midpoint discretization threshold for continuous PSO | not specified in paper |
| PSO empty subset fitness | 1.0 | Worst possible RMSE on binary targets assigned if zero features are selected | not specified in paper |
| PSO subset size penalty | None | Pure RMSE without cardinality penalty; paper reports no feature count regularization | not specified in paper |
| PSO fitness model | DecisionTreeClassifier(random_state=42) | Model evaluating selected subsets unspecified in Section 4.2.3; fast non-linear classifier | not specified in paper |
| PSO cross-validation | 3-fold StratifiedKFold (shuffle=True, random_state=42) | Stratified cross-validation across cleaned dataset | not specified in paper |
| PSO fitness metric | RMSE on hard labels: sqrt(mean((y_true - y_pred)^2)) | Hard predictions; paper reports achieving RMSE 0.3443 after 50 iterations | Section 4.2.3 |
| PSO fitness optimization direction | Minimised as RMSE (equivalent to maximising -RMSE) | Section 4.2.3 text states "search for the minimum objective function value"; Algorithm 1 line 5 '>' treated as typo | Section 4.2.3 |
| PSO global best tracking | Algorithm 1 line 7 ('x**_i = x*_i') read as swarm-wide global best g* | Standard swarm-wide best tracking rather than particle-local re-assignment | Algorithm 1 |
| PSO dimension loop | Algorithm 1 line 9 ('For d = 1 to iteration') read as loop over feature dimensions d = 1..D | Dimension update loop; reading as iterations would create quadratic nesting | Algorithm 1 |
| PSO execution scope (Pipeline A) | Executed on all 9,012 cleaned and scaled accounts before SMOTE and split | Follows Algorithm 6 Step 2 ("feature selection using the PSO algorithm on the whole dataset") | Algorithm 6, Section 4.2 |
| PSO result caching | results/cache/pso_pipeline_a.json keyed by SHA-256 of config + data shape + columns + seed | Deterministic caching with console notification and --force bypass | not specified in paper |
| SMOTE k-neighbors | 5 | Default k_neighbors in imbalanced-learn; Section 4.2.4 names SMOTE but specifies no k-neighbors | not specified in paper |
| SMOTE sampling strategy | "auto" (ratio 1.0, balance minority to majority) | Section 4.2.4 states "creating 7662 samples for both benign and suspicious accounts", resulting in 15,324 rows | Section 4.2.4 |
| SMOTE execution timing (Pipeline A) | Applied to entire dataset of selected features before train/test split | Paper Section 4.2.4 & Figure 6 order: cleaning -> z-score -> PSO -> SMOTE -> split -> models | Section 4.2.4, Figure 6 |
| synthetic sample tracking | Boolean is_synthetic mask tracked through SMOTE and split | Allows measurement of synthetic test set contamination without altering feature arrays | not specified in paper |
| train/test split ratio & stratification | 80/20 train/test split (test_size=0.2, shuffle=True, stratify=None) | Unstratified random split yields exactly 12,259 train and 3,065 test rows; test class split 1,530 / 1,535 is within 14 of paper's 1,544 / 1,521 | Section 4.2.5, Section 5 |
| side-by-side feature sets | Both PSO-selected (22) and Table 2 reference (14) feature sets evaluated | Tests whether paper Table 8 baselines were evaluated on Table 2 features or independent PSO run | not specified in paper |
| Table 2 reference feature resolution | erc20_uniq_sent_addr selected for 14th feature | Table A2 lists only one Unique ERC20 Sent Address feature without the Kaggle CSV's '.1' duplicate; also selected by PSO | Table A2, Table 2 |
| Isolation Forest training scope | Unsupervised fit on X_train only with labels ignored; predicts on X_test | Standard unsupervised anomaly detection; contamination=0.1 from Table 6 | Section 4.3, Table 6 |
| GA population size | 50 | Pinned from text: "population size of 50" | Section 4.3, Algorithm 2 |
| GA generation count | 20 | Pinned from text and Table 4: "over 20 generations" | Section 4.3, Table 4 |
| GA replacement semantics | DEAP eaSimple generational replacement | Algorithm 2 line 8 states "Population = parents + Mutation", but Table 4's N_evals (20-38/gen, 616 total) is consistent with eaSimple (which re-evaluates only changed individuals, no caching), not population doubling | Section 4.3, Algorithm 2, Table 4 |
| GA crossover probability cxpb | 0.5 | DEAP tutorial default; crossover rate not specified in paper | not specified in paper |
| GA mutation probability mutpb | 0.2 | DEAP tutorial default; mutation rate not specified in paper | not specified in paper |
| GA gene mutation probability indpb | 0.2 | Gene-level probability of Gaussian mutation; not specified in paper | not specified in paper |
| GA Gaussian mutation sigma | 10% of gene search range (sigma_fraction=0.1) | Proportional step size for Gaussian perturbation; not specified in paper | not specified in paper |
| GA crossover operator | cxBlend (alpha=0.5) with bounds clipping | Blend crossover for continuous chromosome; not specified in paper | not specified in paper |
| GA selection operator | Roulette-wheel selection (tools.selRoulette) | Section 4.3 explicitly specifies roulette selection; requires strictly positive fitness (> 0) | Section 4.3 |
| GA fitness definition | Stratified 3-fold CV mean accuracy on TRAIN split only | Evaluated on 12,259 train samples; test split strictly excluded to avoid validation leakage | Section 4.3, Section 5 |
| Isolation Forest GA fitness | Unsupervised fit on train folds (labels ignored), evaluated against labels | IF is unsupervised but tuned via supervised accuracy metric per paper Section 3 item 9 | Section 3 Item 9, Section 4.3 |
| XGBoost GA search bounds | n_estimators [10,300], gamma [0,5], min_child_weight [1,10], colsample_bytree [0.3,1.0], max_depth [3,10], reg_lambda [0,10], learning_rate [0.01,1.0]; subsample=1.0 | Search space bounds not given in text; bounds chosen to contain paper Table 5 tuned values | Table 1, Table 5 |
| SVM GA search bounds | Log10-scaled genes: C in [-2, 2] (0.01 to 100), gamma in [-3, 1.301] (0.001 to 20); decoded as 10**gene; kernel='rbf' | Logarithmic scaling covers wide magnitude range; bounds chosen to contain paper Table 6 tuned values | Table 1, Table 6 |
| Isolation Forest GA search bounds | contamination [0.01, 0.5], max_samples [64, 2048], n_estimators [50, 300] | Bounds chosen to contain paper Table 7 tuned values (contamination=0.3, max_samples=1000, n_estimators=89) | Table 1, Table 7 |
| GA integer gene decoding | Round to nearest integer within search bounds | Applies to discrete hyperparameters (n_estimators, max_depth, max_samples) | not specified in paper |
| GA feature set scope | PSO selected features (22 features) by default | Pipeline A applies GA to features selected by PSO (Algorithm 6 Step 5) | Algorithm 6 |
| GA result caching | results/cache/ga_pipeline_a_<model>_<featureset>.json keyed by SHA-256 of config + search space + data shape + columns + seed + versions | Deterministic caching with console notification and --force bypass | not specified in paper |
| CART default hyperparameters | criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42 | Scikit-learn defaults; Saxena et al. [14] CART hyperparameters not specified in paper | Table 10, not specified in paper |
| LOF default hyperparameters | n_neighbors=20, contamination="auto", novelty=True | Scikit-learn defaults; Fangfang et al. [12] LOF hyperparameters not specified in paper; novelty=True required for test evaluation | Table 10, not specified in paper |
| LOF unsupervised training scope | Unsupervised fit on X_train only with labels ignored; predicts on X_test | Standard unsupervised anomaly detection; labels ignored during fit | Table 10, not specified in paper |
| LOF risk score orientation | -score_samples(X_test) (higher = more anomalous/suspicious) | Scikit-learn score_samples produces negative values where lower indicates anomalies; negating aligns scores with suspiciousness for AUC-ROC | not specified in paper |
| baseline feature sets scope | Both PSO-selected (22) and Table 2 (14) evaluated for CART and LOF | Tests whether baselines were evaluated on the paper's selected features or reference set | not specified in paper |
| Table 10 default XGBoost reuse | Reused from Phase 1.5 default XGBoost without retraining | Alarab et al. [16] XGBoost metrics in Table 10 match paper's own default XGBoost in Table 8 | Table 8, Table 10 |
| Table 10 attribution | Attributed to external works ([12], [14], [16]) | Paper Table 10 references external literature for baseline comparison methods | Table 10, Section 5 |

