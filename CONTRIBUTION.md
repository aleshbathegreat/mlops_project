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

### 1. Aleshba Tariq (29222)

- **Tasks & Ownership:**
  - Added `flaml[automl]` to `requirements.txt` and built `src/automl_compare.py` to benchmark bounded FLAML AutoML against manual reference models (`logreg_C10`, `cgpa_only_logreg`) under the exact 80/20 stratified split and `RepeatedStratifiedKFold(5, 3)` protocol.
  - Logged AutoML vs. manual runs (`cv_f1_mean`, `cv_f1_std`, `test_f1`, `test_roc_auc`, `train_time_sec`, `inference_latency_ms`) to MLflow (`admission-automl-comparison`) and generated `reports/automl_comparison.md` and `reports/automl_comparison.json`.
  - Wrote fast unit tests in `tests/test_automl_compare.py` (`--time-budget 2`) and authored the Milestone 2 Project Focus, `CONTRIBUTION.md`, and `AI_USAGE.md` documentation.
- **Reflection:**
  Benchmarking FLAML against our manual champion reinforced that model complexity rarely beats simple linear baselines on small, clean tabular datasets (400 rows). Writing the fair CV comparison and MLflow logging also clarified the trade-off between search cost and negligible metric gains when serving in production.

### 2. Syed M Abdullah Yousuf (29067)

- **Tasks & Ownership:**
  - Implemented thread-safe request monitoring (`threading.Lock`, `time.perf_counter()`), structured stdout logging (`admission_api`), and `GET /stats` in `src/app.py`, along with unit/endpoint tests in `tests/test_app.py`.
  - Hardened `Dockerfile` (`HEALTHCHECK` on `127.0.0.1:8000/health`), `.dockerignore` exclusions, and `.github/workflows/ci.yml` (full-repo Ruff checks and `/health` + `/stats` Docker smoke tests with clean container teardown).
  - Created `RUNBOOK.md` and documented EC2 Docker deployment (port `8000`, optional `/mnt/mlops-data` mount), verification, and rollback procedures in `README.md`.
- **Reflection:**
  Adding in-process `/stats` middleware and container health checks made it straightforward to observe live latency and request counts without slowing down predictions. Separating operational runbook steps from CI also kept routine builds fast while giving us deterministic redeploy and rollback procedures for EC2.
