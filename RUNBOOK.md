# Operations Runbook: Graduate Admission Prediction API

This runbook provides copy-pasteable procedures for building, redeploying, rolling back, and verifying the `admission-api` Docker service locally or on an EC2 host.

---

## 1. Prerequisites & Conventions

- **Service name**: `admission-api`
- **Container port / host port**: `8000:8000`
- **Health endpoint**: `GET /health` (also used by the Dockerfile `HEALTHCHECK` every 30s)
- **Monitoring endpoint**: `GET /stats`
- **Prediction endpoint**: `POST /predict`
- **Image tagging**: Always tag releases with an immutable tag (such as the short Git SHA or a timestamp) in addition to `latest` so rollbacks are deterministic.

```bash
export IMAGE_NAME=admission-api
export CONTAINER_NAME=admission-api
export NEW_TAG="$(git rev-parse --short HEAD)-$(date +%Y%m%d%H%M)"
```

---

## 2. Build, Tag, and Redeploy

### 2.1 Record the currently running image tag (for rollback)

Before replacing the running container, record its current image tag:

```bash
export PREV_IMAGE="$(docker inspect --format='{{.Config.Image}}' "$CONTAINER_NAME" 2>/dev/null || true)"
echo "Previous image: ${PREV_IMAGE:-none}"
```

### 2.2 Pre-build lint & test check (recommended)

```bash
ruff check . && ruff format --check . && pytest tests/ -v
```

### 2.3 Build and tag the new Docker image

The `Dockerfile` copies `data/Large_Dataset.csv`, prepares `data/processed.csv` (2,999 unique records), and trains `models/model.pkl` (`src/prepare_data.py && src/train.py`) during build time so the image is self-contained and reproducible without running slow AutoML searches.

```bash
git pull origin main
docker build -t "${IMAGE_NAME}:${NEW_TAG}" -t "${IMAGE_NAME}:latest" .
docker images "$IMAGE_NAME"
```

### 2.4 Redeploy (Standard self-contained image)

Stop the old container and start the newly tagged image on port `8000`:

```bash
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p 8000:8000 \
  "${IMAGE_NAME}:${NEW_TAG}"
```

### 2.5 Redeploy on EC2 with Persistent `/mnt/mlops-data` Mount (Optional)

If your EC2 instance has an attached EBS/persistent volume mounted at `/mnt/mlops-data`, use it to store datasets, trained model artifacts, and the MLflow tracking database outside the container lifecycle.

1. **Prepare host directories (one-time setup on EC2):**

```bash
sudo mkdir -p /mnt/mlops-data/{data,models,mlflow,reports}
sudo chown -R "$USER:$USER" /mnt/mlops-data
cp -n data/Large_Dataset.csv /mnt/mlops-data/data/Large_Dataset.csv
```

2. **Seed or retrain the model into `/mnt/mlops-data` before mounting `/app/models`:**

> **Note:** `src/app.py` loads `models/model.pkl` at startup. Ensure `/mnt/mlops-data/models/model.pkl` exists before bind-mounting `/mnt/mlops-data/models:/app/models:ro`.

```bash
docker run --rm \
  -v /mnt/mlops-data/data:/app/data \
  -v /mnt/mlops-data/models:/app/models \
  -v /mnt/mlops-data/mlflow:/app/mlflow \
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db \
  "${IMAGE_NAME}:${NEW_TAG}" \
  sh -c "python src/prepare_data.py && python src/train.py"
```

3. **Run the API container backed by `/mnt/mlops-data`:**

```bash
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /mnt/mlops-data/data:/app/data:ro \
  -v /mnt/mlops-data/models:/app/models:ro \
  -v /mnt/mlops-data/mlflow:/app/mlflow \
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db \
  "${IMAGE_NAME}:${NEW_TAG}"
```

4. **Optional: Run AutoML comparison on demand (never inside `docker build` or routine CI):**

```bash
docker run --rm \
  -v /mnt/mlops-data/data:/app/data \
  -v /mnt/mlops-data/mlflow:/app/mlflow \
  -v /mnt/mlops-data/reports:/app/reports \
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db \
  "${IMAGE_NAME}:${NEW_TAG}" \
  python src/automl_compare.py --time-budget 20 --output-dir /app/reports
```

---

## 3. Post-Deploy Verification

Run these checks after every deploy or rollback.

### 3.1 Confirm Docker `HEALTHCHECK` reports `(healthy)`

```bash
docker ps --filter "name=${CONTAINER_NAME}"
docker inspect --format='{{.Name}} -> {{.State.Status}} ({{.State.Health.Status}})' "$CONTAINER_NAME"
```

Wait up to 30–40 seconds after startup (`--start-period=10s`, `--interval=30s`) until `STATUS` in `docker ps` shows `Up ... (healthy)`.

### 3.2 Verify `GET /health`

```bash
curl -fsS http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### 3.3 Check baseline `GET /stats`, invoke `POST /predict`, and confirm `/stats` increments

```bash
# 1. Baseline stats before prediction
curl -fsS http://127.0.0.1:8000/stats && echo

# 2. Send a valid prediction request
curl -fsS -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gre": 320,
    "toefl": 110,
    "university_rating": 4,
    "sop": 4.0,
    "lor": 4.0,
    "cgpa": 9.0,
    "research": 1
  }' && echo

# 3. Re-check stats to verify request counts incremented and latency is recorded
curl -fsS http://127.0.0.1:8000/stats && echo
```

Verify that:
- `POST /predict` returns HTTP `200` with `"admitted"` (`0` or `1`) and `"probability"` (`0.0`–`1.0`).
- On the second `GET /stats` call:
  - `predict_requests` increased by `1`.
  - `total_requests` increased by at least `2` (covering the first `/stats` call and the `/predict` call).
  - `avg_latency_ms` is a positive float (`> 0.0`).
  - `uptime_seconds` is non-negative and increasing.

### 3.4 Inspect live request logs

```bash
docker logs --tail 20 "$CONTAINER_NAME"
```

You should see structured `admission_api` entries for startup, requests, predictions, and any validation/error events:

```text
2026-11-27 18:00:00,010 INFO admission_api - event=startup model_path=models/model.pkl model_type=Pipeline status=ready
2026-11-27 18:00:01,123 INFO admission_api - event=request method=GET path=/health status=200 latency_ms=0.31
2026-11-27 18:00:05,454 INFO admission_api - event=prediction admitted=1 probability=0.9214 gre=320.0 toefl=110.0 cgpa=9.00
2026-11-27 18:00:05,456 INFO admission_api - event=request method=POST path=/predict status=200 latency_ms=1.84
2026-11-27 18:00:06,012 INFO admission_api - event=request method=GET path=/stats status=200 latency_ms=0.19
```

---

## 4. Rollback to a Previous Known-Good Image Tag

If the new deployment fails health checks, returns unexpected errors, or degrades latency:

### 4.1 Inspect logs from the failing deployment and list known-good tags

```bash
docker logs --tail 100 "$CONTAINER_NAME"
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}"
```

### 4.2 Roll back to `PREV_TAG`

Set `PREV_TAG` to the last known-good tag (or use `PREV_IMAGE` captured in Section 2.1):

```bash
export PREV_TAG="<known-good-tag>"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p 8000:8000 \
  "${IMAGE_NAME}:${PREV_TAG}"
```

If deploying on EC2 with `/mnt/mlops-data`:

```bash
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /mnt/mlops-data/data:/app/data:ro \
  -v /mnt/mlops-data/models:/app/models:ro \
  -v /mnt/mlops-data/mlflow:/app/mlflow \
  -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow/mlflow.db \
  "${IMAGE_NAME}:${PREV_TAG}"
```

### 4.3 Verify rollback success

Re-run Section 3 (`docker ps`, `GET /health`, `POST /predict`, and `GET /stats`) and confirm the running container image matches `${IMAGE_NAME}:${PREV_TAG}`:

```bash
docker inspect --format='Running image: {{.Config.Image}}' "$CONTAINER_NAME"
```

---

## 5. Quick Troubleshooting

| Symptom | Check / Fix |
|---|---|
| Container stuck in `(starting)` or `(unhealthy)` | Run `docker logs admission-api` and `docker inspect --format='{{json .State.Health}}' admission-api` to see health probe output. |
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | Find and stop the conflicting process/container (`docker ps` or `ss -tlnp \| grep 8000`). |
| `FileNotFoundError: models/model.pkl` with `/mnt/mlops-data` mount | Run the Section 2.5 step 2 training command so `/mnt/mlops-data/models/model.pkl` is populated before starting the API. |
| EC2 external curl times out on port `8000` | Verify `curl http://127.0.0.1:8000/health` works locally on EC2, then check the EC2 Security Group allows inbound TCP `8000`. |
