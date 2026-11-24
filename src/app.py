import logging
import pickle
import sys
import threading
import time

from fastapi import FastAPI, Request
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

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

_start_time = time.perf_counter()
_stats_lock = threading.Lock()
_total_requests = 0
_predict_requests = 0
_total_latency_ms = 0.0


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
    global _total_requests, _predict_requests, _total_latency_ms
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
    uptime = time.perf_counter() - _start_time
    return {
        "total_requests": total,
        "predict_requests": predict_count,
        "avg_latency_ms": round(avg_latency, 4),
        "uptime_seconds": round(uptime, 2),
    }


@app.post("/predict")
def predict(a: Applicant):
    x = [[a.gre, a.toefl, a.university_rating, a.sop, a.lor, a.cgpa, a.research]]
    return {
        "admitted": int(model.predict(x)[0]),
        "probability": float(model.predict_proba(x)[0][1]),
    }
