import os
import pickle

import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]

CONFIGS = [
    ("logreg", {"C": 0.1}),
    ("logreg", {"C": 1.0}),
    ("logreg", {"C": 10.0}),
    ("rf", {"n_estimators": 100, "max_depth": 3}),
    ("rf", {"n_estimators": 200, "max_depth": 5}),
    ("gb", {"n_estimators": 100, "max_depth": 2}),
]


def build(name, params):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, **params)
        )
    if name == "rf":
        return RandomForestClassifier(random_state=42, **params)
    return GradientBoostingClassifier(random_state=42, **params)


def train():
    df = pd.read_csv("data/processed.csv")
    X, y = df[FEATURES].values, df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    mlflow.set_experiment("admission-prediction")
    best = (-1.0, None, None)
    for name, params in CONFIGS:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_params(params)
            model = build(name, params)
            cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="f1").mean()
            model.fit(X_tr, y_tr)
            mlflow.log_metric("cv_f1", cv_f1)
            mlflow.log_metric("test_f1", f1_score(y_te, model.predict(X_te)))
            proba = model.predict_proba(X_te)[:, 1]
            mlflow.log_metric("test_auc", roc_auc_score(y_te, proba))
            if cv_f1 > best[0]:
                best = (cv_f1, name, model)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best[2], f)
    print(f"Best by CV F1: {best[1]} ({best[0]:.4f})")


if __name__ == "__main__":
    train()
