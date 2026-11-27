# AutoML (FLAML) vs. Manual Training Comparison

## Experimental Setup

- **Dataset**: `data/processed.csv` (2999 rows, target: `Chance of Admit >= 0.75`, about 79% admitted)
- **Split**: Stratified 80/20 (`2399` train / `600` sealed test, `random_state=42`)
- **Cross-Validation**: RepeatedStratifiedKFold(n_splits=5, n_repeats=2) (training split only)
- **FLAML Time Budget**: `60.0s` (internal 5-fold stratified CV optimizing `roc_auc`; the discovered estimator is re-scored with the same repeated CV as the manual models)
- **MLflow Experiment**: `admission-automl-comparison-3k`

## Benchmark Results

| Model | Method | Best Estimator | CV ROC-AUC (mean +/- std) | CV Balanced Acc | Test ROC-AUC | Test Balanced Acc | Fit Time (s) | Latency (ms/sample) |
|---|---|---|---|---|---|---|---|---|
| `svm_rbf_C10` | manual_champion | `svm` | 0.9583 +/- 0.0116 | 0.8914 | 0.9693 | 0.9056 | 1.01 | 1.1792 |
| `logreg_C10` | manual_simple | `logreg` | 0.9232 +/- 0.0213 | 0.8286 | 0.9350 | 0.8510 | 0.02 | 0.8668 |
| `cgpa_only_logreg` | manual_baseline | `cgpa_only` | 0.8923 +/- 0.0172 | 0.7898 | 0.8997 | 0.8009 | 0.01 | 1.7441 |
| `flaml_automl` | automl | `xgboost` | 0.9558 +/- 0.0117 | 0.8733 | 0.9640 | 0.8635 | 60.95 | 15.7207 |

## FLAML Best Configuration

- **Best Estimator**: `xgboost`
- **Hyperparameters**:
```json
{
  "colsample_bylevel": 0.9367708357689092,
  "colsample_bytree": 1.0,
  "learning_rate": 0.10022442139383235,
  "max_leaves": 12,
  "min_child_weight": 0.22161786383500734,
  "n_estimators": 329,
  "reg_alpha": 0.0060858338245746825,
  "reg_lambda": 12.330433910291438,
  "subsample": 0.831815573870732
}
```

## Conclusion: Does AutoML Beat Manual Models on 2999 Records?

On 2999 records, FLAML (xgboost, CV ROC-AUC 0.9558 +/- 0.0117) does not meaningfully outperform the manual champion svm_rbf_C10 (CV ROC-AUC 0.9583 +/- 0.0116) (delta -0.0025, within 1 SD). Both clearly beat the simple logreg_C10 (0.9232) and the CGPA-only baseline (0.8923). Fit times cover the final fit only (1.01s manual vs 60.95s AutoML); the manual champion was chosen after a longer model search, so the development-time comparison is in reports/manual_vs_automl.md.
