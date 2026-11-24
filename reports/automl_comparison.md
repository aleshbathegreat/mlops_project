# AutoML (FLAML) vs. Manual Training Comparison

## Experimental Setup

- **Dataset**: `data/processed.csv` (400 rows, target: `Chance of Admit >= 0.75`)
- **Split**: Stratified 80/20 (`320` train / `80` sealed test, `random_state=42`)
- **Cross-Validation**: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) (evaluated on training split only)
- **FLAML Time Budget**: `20.0s` (internal 5-fold stratified CV optimizing `f1`; discovered estimator re-scored with the exact 5x3 repeated CV protocol)
- **MLflow Experiment**: `admission-automl-comparison`

## Benchmark Results

| Model | Method | Best Estimator | CV F1 (mean +/- std) | Test F1 | Test ROC-AUC | Train Time (s) | Inference Latency (ms/sample) |
|---|---|---|---|---|---|---|---|
| `logreg_C10` | manual_champion | `logreg` | 0.8510 +/- 0.0372 | 0.8406 | 0.9539 | 0.0027 | 0.1872 |
| `cgpa_only_logreg` | manual_baseline | `cgpa_only` | 0.8566 +/- 0.0302 | 0.8406 | 0.9571 | 0.0046 | 0.6917 |
| `flaml_automl` | automl | `xgboost` | 0.8328 +/- 0.0417 | 0.8529 | 0.9498 | 20.0523 | 0.4130 |

## FLAML Best Configuration

- **Best Estimator**: `xgboost`
- **Hyperparameters**:
```json
{
  "colsample_bylevel": 1.0,
  "colsample_bytree": 1.0,
  "learning_rate": 0.09999999999999995,
  "max_leaves": 4,
  "min_child_weight": 0.9999999999999993,
  "n_estimators": 4,
  "reg_alpha": 0.0009765625,
  "reg_lambda": 1.0,
  "subsample": 1.0
}
```

## Conclusion: Does AutoML Outperform Simple/Manual Models on 400 Records?

On 400 records, FLAML (xgboost, CV F1 0.8328 +/- 0.0417) does not meaningfully outperform the simple manual champion logreg_C10 (CV F1 0.8510 +/- 0.0372; delta -0.0182, within 1 SD) or the CGPA-only baseline (CV F1 0.8566). Keeping logreg_C10 is preferable for simplicity, lower training cost (0.0027s vs 20.05s), and fast inference.
