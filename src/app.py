import pickle

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Graduate Admission Predictor")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)


class Applicant(BaseModel):
    gre: float = Field(ge=260, le=340)
    toefl: float = Field(ge=0, le=120)
    university_rating: int = Field(ge=1, le=5)
    sop: float = Field(ge=1, le=5)
    lor: float = Field(ge=1, le=5)
    cgpa: float = Field(ge=0, le=10)
    research: int = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(a: Applicant):
    x = [[a.gre, a.toefl, a.university_rating, a.sop, a.lor, a.cgpa, a.research]]
    return {
        "admitted": int(model.predict(x)[0]),
        "probability": float(model.predict_proba(x)[0][1]),
    }
