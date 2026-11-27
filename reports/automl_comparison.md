# AutoML (FLAML) vs. Manual Training Comparison

## Experimental Setup

- **Dataset**: `data/processed.csv` (2999 rows, target: `Chance of Admit >= 0.75`)
- **Split**: Stratified 80/20 (`2399` train / `600` sealed test, `random_state=42`)
- **Cross-Validation**: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) (evaluated on training split only)
- **FLAML Time Budget**: `20.0s` (internal 5-fold stratified CV optimizing `f1`; discovered estimator re-scored with the exact 5x3 repeated CV protocol)
- **MLflow Experiment**: `admission-automl-comparison`

## Benchmark Results

| Model | Method | Best Estimator | CV F1 (mean +/- std) | Test F1 | Test ROC-AUC | Train Time (s) | Inference Latency (ms/sample) |
|---|---|---|---|---|---|---|---|
| `logreg_C10` | manual_champion | `logreg` | 0.9485 +/- 0.0094 | 0.9458 | 0.9347 | 0.0030 | 0.1877 |
| `cgpa_only_logreg` | manual_baseline | `cgpa_only` | 0.9277 +/- 0.0087 | 0.9262 | 0.8997 | 0.0024 | 0.5520 |
| `flaml_automl` | automl | `lgbm` | 0.9558 +/- 0.0068 | 0.9570 | 0.9612 | 20.1079 | 0.7120 |

## FLAML Best Configuration

- **Best Estimator**: `lgbm`
- **Hyperparameters**:
```json
{
  "colsample_bytree": 0.6622904059772807,
  "learning_rate": 0.31379655220144437,
  "log_max_bin": 6,
  "min_child_samples": 71,
  "n_estimators": 28,
  "num_leaves": 12,
  "reg_alpha": 0.010767168653883007,
  "reg_lambda": 1.3796769826043216
}
```

## Conclusion: Does AutoML Outperform Simple/Manual Models on 2999 Records?

On 2999 records, FLAML (lgbm, CV F1 0.9558 +/- 0.0068) does not meaningfully outperform the simple manual champion logreg_C10 (CV F1 0.9485 +/- 0.0094; delta +0.0073, within 1 SD) or the CGPA-only baseline (CV F1 0.9277). Keeping logreg_C10 is preferable for simplicity, lower training cost (0.0030s vs 20.11s), and fast inference.
