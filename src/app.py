import logging
import pickle
import sys
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MODEL_PATH = "models/model.pkl"

logger = logging.getLogger("admission_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logger.addHandler(_handler)
logger.propagate = False

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as exc:
    logger.exception(
        "event=startup status=error model_path=%s error=%s",
        MODEL_PATH,
        type(exc).__name__,
    )
    raise


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "event=startup model_path=%s model_type=%s status=ready",
        MODEL_PATH,
        type(model).__name__,
    )
    yield
    logger.info("event=shutdown status=stopped")


app = FastAPI(title="Graduate Admission Predictor", lifespan=lifespan)

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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        "event=validation_error method=%s path=%s errors=%d",
        request.method,
        request.url.path,
        len(exc.errors()),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    global _total_requests, _predict_requests, _total_latency_ms
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        status_code = 500
        logger.exception(
            "event=error method=%s path=%s error=%s",
            request.method,
            request.url.path,
            type(exc).__name__,
        )
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        with _stats_lock:
            _total_requests += 1
            if request.method == "POST" and request.url.path == "/predict":
                _predict_requests += 1
            _total_latency_ms += latency_ms
        logger.info(
            "event=request method=%s path=%s status=%d latency_ms=%.2f",
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
    admitted = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][1])
    logger.info(
        "event=prediction admitted=%d probability=%.4f gre=%.1f toefl=%.1f cgpa=%.2f",
        admitted,
        probability,
        a.gre,
        a.toefl,
        a.cgpa,
    )
    return {
        "admitted": admitted,
        "probability": probability,
    }
