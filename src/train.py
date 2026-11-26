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
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    roc_auc_score,
)
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
CV_SPLITS = 5
CV_REPEATS = 2
CW = "balanced"  # the admitted class is about 79%, so weight classes equally

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


def scaled(model):
    return make_pipeline(StandardScaler(), model)


def members():
    return [
        ("lr", scaled(LogisticRegression(max_iter=1000, class_weight=CW))),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200, max_depth=5, class_weight=CW, random_state=42
            ),
        ),
        ("svm", scaled(SVC(probability=True, class_weight=CW, random_state=42))),
    ]


def build(family, p):
    if family == "dummy":
        return DummyClassifier(random_state=42, **p)
    if family == "cgpa_only":
        return make_pipeline(
            ColumnTransformer([("cgpa", StandardScaler(), [CGPA_IDX])]),
            LogisticRegression(max_iter=1000, class_weight=CW),
        )
    if family == "logreg":
        return scaled(LogisticRegression(max_iter=1000, class_weight=CW, **p))
    if family == "tree":
        return DecisionTreeClassifier(class_weight=CW, random_state=42, **p)
    if family == "rf":
        return RandomForestClassifier(class_weight=CW, random_state=42, **p)
    if family == "extra":
        return ExtraTreesClassifier(class_weight=CW, random_state=42, **p)
    if family == "gb":
        return GradientBoostingClassifier(random_state=42, **p)
    if family == "svm":
        return scaled(SVC(probability=True, class_weight=CW, random_state=42, **p))
    if family == "knn":
        return scaled(KNeighborsClassifier(**p))
    if family == "mlp":
        return scaled(MLPClassifier(max_iter=2000, random_state=42, **p))
    if family == "voting":
        return VotingClassifier(estimators=members(), voting="soft")
    if family == "stacking":
        return StackingClassifier(
            estimators=members(),
            final_estimator=LogisticRegression(max_iter=1000, class_weight=CW),
            cv=5,
        )
    raise ValueError(f"Unknown family: {family}")


def pick_champion(rows):
    """rows: (name, family, cv_auc, cv_auc_std, cv_bal_acc, test_auc, test_bal_acc).

    Best full-feature model by CV ROC-AUC, then the simplest model within one
    standard deviation of it.
    """
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
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=42
    )
    scoring = {
        "auc": "roc_auc",
        "bal_acc": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1": make_scorer(f1_score, zero_division=0),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("admission-prediction-3k")

    rows = []
    fitted = {}
    for name, family, params in CONFIGS:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("family", family)
            mlflow.log_params({k: str(v) for k, v in params.items()})

            model = build(family, params)
            res = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring)
            cv_auc = res["test_auc"].mean()
            cv_auc_std = res["test_auc"].std()
            cv_bal = res["test_bal_acc"].mean()
            mlflow.log_metric("cv_auc", cv_auc)
            mlflow.log_metric("cv_auc_std", cv_auc_std)
            mlflow.log_metric("cv_bal_acc", cv_bal)
            mlflow.log_metric("cv_f1_macro", res["test_f1_macro"].mean())
            mlflow.log_metric("cv_f1", res["test_f1"].mean())

            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            test_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
            test_bal = balanced_accuracy_score(y_te, pred)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("test_bal_acc", test_bal)
            mlflow.log_metric("test_f1", f1_score(y_te, pred, zero_division=0))

            rows.append((name, family, cv_auc, cv_auc_std, cv_bal, test_auc, test_bal))
            fitted[name] = model
            print(f"{name:20s} cv_auc={cv_auc:.4f} cv_bal_acc={cv_bal:.4f}", flush=True)

    champion = pick_champion(rows)
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(fitted[champion[0]], f)

    print("\nRanking by CV ROC-AUC:")
    for r in sorted(rows, key=lambda r: r[2], reverse=True):
        print(
            f"{r[0]:20s} {r[1]:9s} cv_auc={r[2]:.4f} (+/-{r[3]:.4f}) "
            f"cv_bal_acc={r[4]:.4f} test_auc={r[5]:.4f} test_bal_acc={r[6]:.4f}"
        )
    print(f"\nChampion: {champion[0]} (CV ROC-AUC {champion[2]:.4f})")


if __name__ == "__main__":
    train()
