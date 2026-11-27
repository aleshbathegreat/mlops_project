# Manual ML vs AutoML (FLAML)

**Research question:** can AutoML achieve comparable or better predictive performance than manual machine learning, while reducing model development time and effort?

## Setup

- Dataset: `data/processed.csv` (2999 rows, about 79% admitted)
- Seeds (independent 80/20 stratified splits): [42, 43, 44, 45, 46]
- Manual: 17 model configurations, RepeatedStratifiedKFold(n_splits=5, n_repeats=2), champion = best CV ROC-AUC then simplest within 1 SD
- AutoML: FLAML, ROC-AUC objective, time budgets [15.0, 60.0] s, same balanced class weights as the manual models
- Every model is developed on the training split only and scored once on the sealed test split
- MLflow experiment: `manual-vs-automl-3k`

## Performance and reliability (mean +/- std over seeds)

| Approach | Test ROC-AUC | Test balanced acc | Test F1 |
|---|---|---|---|
| `manual` | 0.9630 +/- 0.0049 | 0.8991 +/- 0.0165 | 0.9470 +/- 0.0077 |
| `automl_15s` | 0.9557 +/- 0.0044 | 0.8747 +/- 0.0148 | 0.9341 +/- 0.0101 |
| `automl_60s` | 0.9603 +/- 0.0042 | 0.8865 +/- 0.0090 | 0.9443 +/- 0.0051 |

## AutoML vs manual, paired by seed (test ROC-AUC)

| AutoML budget | Mean difference | Seeds AutoML won | Verdict |
|---|---|---|---|
| `automl_15s` | -0.0073 +/- 0.0044 | 0/5 | worse |
| `automl_60s` | -0.0027 +/- 0.0025 | 1/5 | worse |

Verdict rule: *better* if the mean difference is more than one SD above zero, *worse* if more than one SD below, otherwise *comparable*.

## Efficiency

| Approach | Development time (s) | Inference (ms/row) | Model size (KB) |
|---|---|---|---|
| `manual` | 256.7 +/- 30.9 | 1.100 +/- 0.114 | 47 +/- 0 |
| `automl_15s` | 15.7 +/- 0.3 | 9.875 +/- 11.053 | 194 +/- 46 |
| `automl_60s` | 60.6 +/- 0.5 | 15.621 +/- 23.380 | 749 +/- 508 |

Effort proxy (lines of code): manual pipeline 222, AutoML call 30. Lines of code ignore the human time spent choosing models and ranges; record your own hours in the README.

## Which model each approach picked

- `manual`: svm_rbf_C10 x5
- `automl_15s`: lgbm x3, rf x1, xgboost x1
- `automl_60s`: lgbm x3, rf x1, xgboost x1

## Conclusion

On 2999 records over 5 seeds. With a 15s budget AutoML was worse (test AUC 0.9557 vs 0.9630; paired difference -0.0073 +/- 0.0044, AutoML ahead in 0/5 seeds) and finished 16.4x faster than the manual search (16s vs 257s). With a 60s budget AutoML was worse (test AUC 0.9603 vs 0.9630; paired difference -0.0027 +/- 0.0025, AutoML ahead in 1/5 seeds) and finished 4.2x faster than the manual search (61s vs 257s).
