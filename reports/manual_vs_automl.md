# Manual Training vs. FLAML AutoML (5-Seed Study on 2,999 Records)

## Experimental Setup

- **Raw Dataset**: `data/Large_Dataset.csv` (3,000 rows, 10 columns)
- **Processed Dataset**: `data/processed.csv` (**2,999 unique records** after dropping `Serial No.`, dropping `Motivation Letter` to preserve the 7-feature numeric `/predict` contract, and deduplicating on the 7 academic features; target: `Chance of Admit >= 0.75`)
- **Evaluation Protocol**: 5 random seeds, stratified 80/20 train/test split (`2,399` train / `600` sealed test) with `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` on the training split
- **Manual Pipeline**: [`src/train.py`](../src/train.py) evaluating 20 configurations across 8 model families (including `SVC` with RBF kernel, linear models, trees, KNN, MLP, and voting/stacking ensembles)
- **AutoML Pipeline**: [`src/automl_compare.py`](../src/automl_compare.py) running bounded `FLAML` `AutoML` (`lgbm`, `xgboost`, `rf`, `extra_tree`, `lrl1`, `lrl2`, `kneighbor`) at `15s` and `60s` search budgets

## 5-Seed Summary Results (2,999 Records)

| Approach | Budget / Dev Time | Mean Test ROC-AUC | Wins vs. Manual (5 Seeds) | Notes |
|---|---|---|---|---|
| **Manual 20-model pipeline** (`src/train.py`) | `257s` total | **0.9630** | Baseline | Includes RBF `SVC`, ensembles, and complexity-aware CV selection |
| **FLAML AutoML (`15s` budget)** | `15s` budget / `16s` dev time | `0.9557` | `0 / 5` | Fast discovery, trails manual pipeline by `-0.0073` ROC-AUC |
| **FLAML AutoML (`60s` budget)** | `60s` budget / `61s` dev time | `0.9603` | `1 / 5` | Closes most of the gap, but still trails manual on 4/5 seeds |

## Key Findings

1. **Manual training wins on 4/5 seeds and overall ROC-AUC (`0.9630`)**: With 2,999 records, the manual 20-model suite achieves higher mean ROC-AUC (`0.9630`) than both 15s FLAML (`0.9557`, `0/5` wins) and 60s FLAML (`0.9603`, `1/5` wins).
2. **FLAML lacks SVM (`SVC`) in its default search space**: On this clean 7-feature numeric boundary, RBF-kernel SVM is a strong candidate in `src/train.py`, whereas FLAML's default tabular estimator list (`lgbm`, `xgboost`, `rf`, `extra_tree`, `lrl1`, `lrl2`, `kneighbor`) does not include kernel SVM.
3. **Time-budget vs. developer-time trade-off**: FLAML is fast to iterate (`16s` at `15s` budget, `61s` at `60s` budget vs. `257s` for the full 5-seed manual sweep), making it valuable for rapid baseline exploration, while the manual pipeline remains preferable for final champion selection and `/predict` deployment.
4. **API contract preserved**: Dropping `Motivation Letter` in `src/prepare_data.py` keeps the 7-feature numeric schema (`GRE Score`, `TOEFL Score`, `University Rating`, `SOP`, `LOR`, `CGPA`, `Research`) intact so `/predict` remains 100% backward-compatible.
