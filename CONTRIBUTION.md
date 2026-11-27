# Team Contributions

## Milestone 1

1. Aleshba Tariq (29222)

Tasks: setup,data preparation, training and MLfLow

Remarks: dataset was small and mostly numerical so a lot of pre-processing was not needed. Hence, focus was laid on training using 20 models (6 different models and their variations) and then the tracking in MLflow.


2. Abdullah Yousuf (29067)

Tasks: API endpoints, tests and Ruff, Docker and CI pipeline

Remarks: creating health, predict and docs endpoints and testing them with different inputs to check for errors and proper implementation of code. Regular merging was done in github to ensure smooth CI pipeline. Both jobs: test and docker succeeded.

---

## Milestone 2

### 1. Aleshba Tariq (29222) (estimated ~12 hours)

- **Tasks & Ownership:**
  - Added `flaml[automl]` to `requirements.txt` and built `src/automl_compare.py` to benchmark bounded FLAML AutoML against manual reference models (`logreg_C10`, `cgpa_only_logreg`) under the exact 80/20 stratified split and `RepeatedStratifiedKFold(5, 3)` protocol.
  - Scaled `src/prepare_data.py` to `data/Large_Dataset.csv` (2,999 unique records after dropping `Serial No.`, `Motivation Letter`, and 1 duplicate on the 7 academic features) while preserving the 7-feature `/predict` contract.
  - Ran the 5-seed manual vs. FLAML AutoML study on 2,999 records (manual `0.9630` ROC-AUC in `257s` vs. FLAML `0.9557` at `15s` / `16s` dev time / `0/5` wins and `0.9603` at `60s` / `61s` dev time / `1/5` wins, noting FLAML lacks RBF `SVC`), logged runs to MLflow (`admission-automl-comparison`), and documented results in `reports/manual_vs_automl.md` and `reports/automl_comparison.*`.
  - Wrote fast unit tests in `tests/test_automl_compare.py` (`--time-budget 2`) and authored the Milestone 2 Project Focus, `CONTRIBUTION.md`, and `AI_USAGE.md` documentation.
- **Reflection:**
  Scaling from 400 to 2,999 records confirmed that a curated manual model suite with RBF `SVC` (`0.9630` ROC-AUC) still beats bounded FLAML on 4/5 seeds, while FLAML excels at fast baseline iteration (`16s–61s` dev time vs. `257s`).

### 2. Syed M Abdullah Yousuf (29067) (estimated ~12 hours)

- **Tasks & Ownership:**
  - Implemented thread-safe request monitoring (`threading.Lock`, `time.perf_counter()`), structured stdout logging (`event=startup`, `event=request`, `event=prediction`, `event=validation_error`, `event=error`), and `GET /stats` in `src/app.py`, along with endpoint and logging tests in `tests/test_app.py`.
  - Hardened `Dockerfile` (`Large_Dataset.csv` build, `HEALTHCHECK` on `127.0.0.1:8000/health`), `.dockerignore` exclusions, and `.github/workflows/ci.yml` (full-repo Ruff checks and `/health`, `/stats`, `/predict` Docker smoke tests with clean container teardown).
  - Updated `RUNBOOK.md` and documented EC2 Docker deployment (port `8000`, optional `/mnt/mlops-data` mount), verification, and rollback procedures in `README.md`.
- **Reflection:**
  Adding structured startup, prediction, validation, and error logs alongside `/stats` middleware made it straightforward to observe live latency, request counts, and failure modes without slowing down predictions. Separating operational runbook steps from CI also kept routine builds fast while giving us deterministic redeploy and rollback procedures for EC2.
