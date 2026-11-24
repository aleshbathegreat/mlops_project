import pickle

import pandas as pd
import pytest
from fastapi.testclient import TestClient

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
    assert len(df) > 0
    assert df.isnull().sum().sum() == 0
    assert "Serial No." not in df.columns
    assert set(df["target"].unique()) <= {0, 1}


def test_prepare_data_rejects_bad_data(tmp_path):
    raw = pd.read_csv("data/Admission_Predict.csv")
    raw.loc[0, "CGPA"] = 15
    bad = tmp_path / "bad.csv"
    raw.to_csv(bad, index=False)
    with pytest.raises(AssertionError):
        prepare_data(raw_path=str(bad), out_path=str(tmp_path / "o.csv"))
