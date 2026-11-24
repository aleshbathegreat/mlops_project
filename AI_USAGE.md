# AI Tool Usage Disclosure

## Tools Used

- **Antigravity IDE Assistant** powered by **Google Gemini** models (used for code drafting, refactoring, test generation, and documentation review across Milestone 2).

## Tasks Supported

1. **Monitoring, Containerization & Endpoint Tests**
   - Assisted with designing thread-safe request-counting and latency middleware (`threading.Lock`, `time.perf_counter()`), structured stdout request logging, and the `GET /stats` endpoint in `src/app.py`.
   - Assisted with refining `Dockerfile` `HEALTHCHECK`, `.dockerignore`, and adding `/stats` verification tests in `tests/test_app.py`.
2. **AutoML vs. Manual Benchmarking**
   - Assisted with drafting `src/automl_compare.py` to run bounded `FLAML` `AutoML` alongside `logreg_C10` and `cgpa_only_logreg` using the exact `RepeatedStratifiedKFold(5, 3)` split from `src/train.py`, MLflow tracking, and report generation (`reports/automl_comparison.md`, `reports/automl_comparison.json`).
   - Assisted with writing isolated, fast unit tests in `tests/test_automl_compare.py`.
3. **CI Pipeline, Operations Runbook & Documentation**
   - Assisted with hardening `.github/workflows/ci.yml` smoke tests and creating `RUNBOOK.md` (EC2 `/mnt/mlops-data` deployment, verification, and image-tag rollback procedures).
   - Assisted with drafting and structuring updates to `README.md`, `CONTRIBUTION.md`, and `AI_USAGE.md`.

## Human Review & Ownership

All team members (Aleshba Tariq and Syed M Abdullah Yousuf) personally reviewed, understood, and verified every AI-assisted file, command, and metric before committing or merging. All changes were independently validated with `ruff check .`, `ruff format --check .`, `pytest tests/ -v`, MLflow run inspection, and Docker/API checks.
