import os
import pickle
import warnings

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

FEATURES = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
]
CGPA_IDX = FEATURES.index("CGPA")

# (run name, model family, hyperparameters)
CONFIGS = [
    # baselines
    ("dummy_majority", "dummy", {"strategy": "most_frequent"}),
    ("dummy_stratified", "dummy", {"strategy": "stratified"}),
    ("cgpa_only_logreg", "cgpa_only", {}),
    # logistic regression
    ("logreg_C0.1", "logreg", {"C": 0.1}),
    ("logreg_C1", "logreg", {"C": 1.0}),
    ("logreg_C10", "logreg", {"C": 10.0}),
    # tree-based
    ("tree_depth3", "tree", {"max_depth": 3}),
    ("tree_depth5", "tree", {"max_depth": 5}),
    ("rf_100_depth3", "rf", {"n_estimators": 100, "max_depth": 3}),
    ("rf_200_depth5", "rf", {"n_estimators": 200, "max_depth": 5}),
    ("extra_200_depth5", "extra", {"n_estimators": 200, "max_depth": 5}),
    ("gb_100_depth2", "gb", {"n_estimators": 100, "max_depth": 2}),
    # SVM
    ("svm_rbf_C1", "svm", {"C": 1.0}),
    ("svm_rbf_C10", "svm", {"C": 10.0}),
    # KNN
    ("knn_k5", "knn", {"n_neighbors": 5}),
    ("knn_k15", "knn", {"n_neighbors": 15}),
    # small neural network
    ("mlp_16", "mlp", {"hidden_layer_sizes": (16,)}),
    ("mlp_32_16", "mlp", {"hidden_layer_sizes": (32, 16)}),
    # ensembles
    ("voting_soft", "voting", {}),
    ("stacking", "stacking", {}),
]


def scaled(model):
    return make_pipeline(StandardScaler(), model)


def members():
    return [
        ("lr", scaled(LogisticRegression(max_iter=1000, C=1.0))),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
        ("svm", scaled(SVC(probability=True, C=1.0, random_state=42))),
    ]


def build(family, p):
    if family == "dummy":
        return DummyClassifier(random_state=42, **p)
    if family == "cgpa_only":
        return make_pipeline(
            ColumnTransformer([("cgpa", StandardScaler(), [CGPA_IDX])]),
            LogisticRegression(max_iter=1000),
        )
    if family == "logreg":
        return scaled(LogisticRegression(max_iter=1000, **p))
    if family == "tree":
        return DecisionTreeClassifier(random_state=42, **p)
    if family == "rf":
        return RandomForestClassifier(random_state=42, **p)
    if family == "extra":
        return ExtraTreesClassifier(random_state=42, **p)
    if family == "gb":
        return GradientBoostingClassifier(random_state=42, **p)
    if family == "svm":
        return scaled(SVC(probability=True, random_state=42, **p))
    if family == "knn":
        return scaled(KNeighborsClassifier(**p))
    if family == "mlp":
        return scaled(MLPClassifier(max_iter=2000, random_state=42, **p))
    if family == "voting":
        return VotingClassifier(estimators=members(), voting="soft")
    if family == "stacking":
        return StackingClassifier(
            estimators=members(),
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
        )
    raise ValueError(f"Unknown family: {family}")


# Baselines are reference points, not deployable champions.
BASELINES = {"dummy", "cgpa_only"}
# Lower = simpler. Breaks ties between models that score about the same.
COMPLEXITY = {
    "logreg": 1,
    "tree": 2,
    "knn": 3,
    "svm": 3,
    "rf": 4,
    "extra": 4,
    "gb": 4,
    "mlp": 5,
    "voting": 6,
    "stacking": 6,
}


def pick_champion(rows):
    """rows: (name, family, cv_f1, cv_f1_std, test_f1, test_auc)."""
    eligible = [r for r in rows if r[1] not in BASELINES]
    top = max(eligible, key=lambda r: r[2])
    close = [r for r in eligible if r[2] >= top[2] - top[3]]
    return min(close, key=lambda r: (COMPLEXITY[r[1]], -r[2]))


def train():
    df = pd.read_csv("data/processed.csv")
    X, y = df[FEATURES].values, df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scoring = {
        "f1": make_scorer(f1_score, zero_division=0),
        "auc": "roc_auc",
        "acc": "accuracy",
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("admission-prediction")

    rows = []
    fitted = {}
    for name, family, params in CONFIGS:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("family", family)
            mlflow.log_params({k: str(v) for k, v in params.items()})

            model = build(family, params)
            res = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring)
            cv_f1 = res["test_f1"].mean()
            cv_auc = res["test_auc"].mean()
            mlflow.log_metric("cv_f1", cv_f1)
            mlflow.log_metric("cv_f1_std", res["test_f1"].std())
            mlflow.log_metric("cv_auc", cv_auc)
            mlflow.log_metric("cv_acc", res["test_acc"].mean())

            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            proba = model.predict_proba(X_te)[:, 1]
            test_f1 = f1_score(y_te, pred, zero_division=0)
            test_auc = roc_auc_score(y_te, proba)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("test_acc", accuracy_score(y_te, pred))

            rows.append((name, family, cv_f1, res["test_f1"].std(), test_f1, test_auc))
            print(f"{name:20s} cv_f1={cv_f1:.4f} test_f1={test_f1:.4f}")

            # champion is chosen on cross-validation only, never on the test set
            fitted[name] = model

    champion = pick_champion(rows)
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(fitted[champion[0]], f)

    print("\nRanking by CV F1:")
    for r in sorted(rows, key=lambda r: r[2], reverse=True):
        print(
            f"{r[0]:20s} {r[1]:9s} cv_f1={r[2]:.4f} (+/-{r[3]:.4f}) "
            f"test_f1={r[4]:.4f} test_auc={r[5]:.4f}"
        )
    print(f"\nChampion: {champion[0]} (CV F1 {champion[2]:.4f})")


if __name__ == "__main__":
    train()
