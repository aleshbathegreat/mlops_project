# AI Tool Usage Disclosure

## Tools Used

- **Antigravity IDE Assistant** powered by **Google Gemini** models (used for code drafting, refactoring, test generation, and documentation review across Milestone 2).

## Tasks Supported

1. **Monitoring, Structured Logging, Containerization & Endpoint Tests**
   - Assisted with designing thread-safe request-counting and latency middleware (`threading.Lock`, `time.perf_counter()`), structured stdout logging (`event=startup`, `event=request`, `event=prediction`, `event=validation_error`, `event=error`), and `GET /stats` in `src/app.py`.
   - Assisted with refining `Dockerfile` (`Large_Dataset.csv`, `HEALTHCHECK`), `.dockerignore`, and adding `/stats` and structured logging verification tests in `tests/test_app.py`.
2. **2,999-Record Data Alignment & AutoML vs. Manual Benchmarking**
   - Assisted with updating `src/prepare_data.py` to process `data/Large_Dataset.csv` into 2,999 unique records while preserving the 7-feature `/predict` contract, and updating `src/automl_compare.py` and `reports/manual_vs_automl.md` with the 5-seed benchmark findings.
   - Assisted with writing and updating fast unit tests in `tests/test_automl_compare.py`.
3. **CI Pipeline, Operations Runbook & Documentation**
   - Assisted with hardening `.github/workflows/ci.yml` smoke tests and creating `RUNBOOK.md` (EC2 `/mnt/mlops-data` deployment, verification, and image-tag rollback procedures).
   - Assisted with drafting and structuring updates to `README.md`, `CONTRIBUTION.md`, and `AI_USAGE.md`.

## Human Review & Ownership

All team members (Aleshba Tariq and Syed M Abdullah Yousuf) personally reviewed, understood, and verified every AI-assisted file, command, and metric before committing or merging. All changes were independently validated with `ruff check .`, `ruff format --check .`, `pytest tests/ -v`, MLflow run inspection, and Docker/API checks.
