import json
import logging
import pickle
import sys
import threading
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("admission_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logger.addHandler(_handler)
logger.propagate = False

app = FastAPI(title="Graduate Admission Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
logger.info("Loaded model %s from models/model.pkl", type(model).__name__)

_start_time = time.perf_counter()
_stats_lock = threading.Lock()
_total_requests = 0
_predict_requests = 0
_total_latency_ms = 0.0
_error_responses = 0
_predicted_admitted = 0
_predict_latencies = deque(maxlen=1000)  # ms, most recent /predict calls


class Applicant(BaseModel):
    gre: float = Field(ge=260, le=340)
    toefl: float = Field(ge=0, le=120)
    university_rating: int = Field(ge=1, le=5)
    sop: float = Field(ge=1, le=5)
    lor: float = Field(ge=1, le=5)
    cgpa: float = Field(ge=0, le=10)
    research: int = Field(ge=0, le=1)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    global _total_requests, _predict_requests, _total_latency_ms, _error_responses
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        with _stats_lock:
            _total_requests += 1
            if request.method == "POST" and request.url.path == "/predict":
                _predict_requests += 1
                _predict_latencies.append(latency_ms)
            if status_code >= 400:
                _error_responses += 1
            _total_latency_ms += latency_ms
        logger.info(
            "%s %s -> %d (%.2f ms)",
            request.method,
            request.url.path,
            status_code,
            latency_ms,
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    with _stats_lock:
        total = _total_requests
        predict_count = _predict_requests
        avg_latency = (_total_latency_ms / total) if total > 0 else 0.0
        errors = _error_responses
        predicted = _predicted_admitted
        lat = sorted(_predict_latencies)
    predict_avg = (sum(lat) / len(lat)) if lat else 0.0
    predict_p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
    admit_rate = (predicted / predict_count) if predict_count > 0 else 0.0
    uptime = time.perf_counter() - _start_time
    return {
        "total_requests": total,
        "predict_requests": predict_count,
        "avg_latency_ms": round(avg_latency, 4),
        "error_responses": errors,
        "predict_avg_latency_ms": round(predict_avg, 4),
        "predict_p95_latency_ms": round(predict_p95, 4),
        "predicted_admit_rate": round(admit_rate, 4),
        "uptime_seconds": round(uptime, 2),
    }


@app.post("/predict")
def predict(a: Applicant):
    x = [[a.gre, a.toefl, a.university_rating, a.sop, a.lor, a.cgpa, a.research]]
    admitted = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][1])
    global _predicted_admitted
    with _stats_lock:
        _predicted_admitted += admitted
    logger.info(
        "prediction %s",
        json.dumps(
            {
                "inputs": a.model_dump(),
                "admitted": admitted,
                "probability": round(probability, 4),
            }
        ),
    )
    return {"admitted": admitted, "probability": probability}
