import logging
import pickle

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.app as app_module
from src.app import app
from src.prepare_data import prepare_data

client = TestClient(app)

VALID = {
    "gre": 320,
    "toefl": 110,
    "university_rating": 4,
    "sop": 4.0,
    "lor": 4.0,
    "cgpa": 9.0,
    "research": 1,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid_input():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["admitted"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_wrong_type():
    r = client.post("/predict", json={**VALID, "gre": "abc"})
    assert r.status_code == 422


def test_predict_out_of_range():
    r = client.post("/predict", json={**VALID, "cgpa": 15})
    assert r.status_code == 422


def test_model_loads_and_predicts():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model.predict([[320, 110, 4, 4, 4, 9, 1]])[0] in (0, 1)


def test_prepare_data_output(tmp_path):
    df = prepare_data(out_path=str(tmp_path / "p.csv"))
    assert len(df) == 2999
    assert df.isnull().sum().sum() == 0
    assert "Serial No." not in df.columns
    assert "Motivation Letter" not in df.columns
    assert set(df["target"].unique()) <= {0, 1}


def test_prepare_data_rejects_bad_data(tmp_path):
    raw = pd.read_csv("data/Large_Dataset.csv")
    raw.loc[0, "CGPA"] = 15
    bad = tmp_path / "bad.csv"
    raw.to_csv(bad, index=False)
    with pytest.raises(AssertionError):
        prepare_data(raw_path=str(bad), out_path=str(tmp_path / "o.csv"))


def test_stats_endpoint():
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "total_requests",
        "predict_requests",
        "avg_latency_ms",
        "uptime_seconds",
    }
    assert isinstance(body["total_requests"], int) and body["total_requests"] >= 0
    assert isinstance(body["predict_requests"], int) and body["predict_requests"] >= 0
    assert isinstance(body["avg_latency_ms"], float) and body["avg_latency_ms"] >= 0.0
    assert isinstance(body["uptime_seconds"], float) and body["uptime_seconds"] >= 0.0


def test_stats_tracks_predict_requests():
    before = client.get("/stats").json()
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    after = client.get("/stats").json()
    assert after["predict_requests"] == before["predict_requests"] + 1
    assert after["total_requests"] == before["total_requests"] + 2
    assert after["avg_latency_ms"] > 0.0
    assert after["uptime_seconds"] >= before["uptime_seconds"]


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture()
def app_logs():
    logger = logging.getLogger("admission_api")
    handler = _ListHandler()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


def test_startup_and_shutdown_logging(app_logs):
    with TestClient(app) as local_client:
        r = local_client.get("/health")
        assert r.status_code == 200
    messages = [rec.getMessage() for rec in app_logs]
    assert any(
        "event=startup" in m
        and "model_path=models/model.pkl" in m
        and "status=ready" in m
        for m in messages
    )
    assert any("event=shutdown" in m and "status=stopped" in m for m in messages)


def test_request_and_prediction_logging(app_logs):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    messages = [rec.getMessage() for rec in app_logs]
    assert any(
        "event=prediction" in m and "admitted=" in m and "probability=" in m
        for m in messages
    )
    assert any(
        "event=request" in m
        and "method=POST" in m
        and "path=/predict" in m
        and "status=200" in m
        for m in messages
    )


def test_validation_error_logging(app_logs):
    r = client.post("/predict", json={**VALID, "cgpa": 15})
    assert r.status_code == 422
    messages = [rec.getMessage() for rec in app_logs]
    assert any(
        "event=validation_error" in m and "method=POST" in m and "path=/predict" in m
        for m in messages
    )
    assert any(
        "event=request" in m and "path=/predict" in m and "status=422" in m
        for m in messages
    )


def test_unhandled_error_logging(app_logs, monkeypatch):
    class _BrokenModel:
        def predict(self, _x):
            raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "model", _BrokenModel())
    failing_client = TestClient(app, raise_server_exceptions=False)
    r = failing_client.post("/predict", json=VALID)
    assert r.status_code == 500
    messages = [rec.getMessage() for rec in app_logs]
    assert any(
        "event=error" in m and "path=/predict" in m and "error=RuntimeError" in m
        for m in messages
    )
    assert any(
        "event=request" in m and "path=/predict" in m and "status=500" in m
        for m in messages
    )
