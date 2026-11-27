import argparse
import importlib.util
import json
import os
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.train import FEATURES, build
except ModuleNotFoundError:
    from train import FEATURES, build

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = "data/processed.csv"
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_TIME_BUDGET = 20.0
DEFAULT_RANDOM_STATE = 42
EXPERIMENT_NAME = "admission-automl-comparison"


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _available_flaml_estimators():
    estimators = ["rf", "extra_tree", "lrl1", "lrl2", "kneighbor"]
    if importlib.util.find_spec("xgboost") is not None:
        estimators.insert(0, "xgboost")
    if importlib.util.find_spec("lightgbm") is not None:
        estimators.insert(0, "lgbm")
    return estimators


def measure_inference_latency_ms(model, X_te):
    start = time.perf_counter()
    for row in X_te:
        sample = row.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            model.predict_proba(sample)
        else:
            model.predict(sample)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return float(elapsed_ms / len(X_te))


def _evaluate_manual_model(name, method, family, params, X_tr, y_tr, X_te, y_te, cv):
    f1_scorer = make_scorer(f1_score, zero_division=0)
    model = build(family, params)
    cv_res = cross_validate(model, X_tr, y_tr, cv=cv, scoring={"f1": f1_scorer})

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time_sec = time.perf_counter() - t0

    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    latency_ms = measure_inference_latency_ms(model, X_te)

    return {
        "name": name,
        "method": method,
        "best_estimator": family,
        "best_config": _to_json_safe(params),
        "cv_f1_mean": round(float(cv_res["test_f1"].mean()), 4),
        "cv_f1_std": round(float(cv_res["test_f1"].std()), 4),
        "test_f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "train_time_sec": round(float(train_time_sec), 4),
        "inference_latency_ms": round(float(latency_ms), 4),
    }


def _evaluate_flaml_cv(automl, X_tr, y_tr, cv):
    f1_scorer = make_scorer(f1_score, zero_division=0)
    estimator = automl.model.estimator
    try:
        est_for_cv = clone(estimator)
    except (TypeError, ValueError):
        est_for_cv = estimator

    if str(automl.best_estimator) in {"lrl1", "lrl2"}:
        est_for_cv = make_pipeline(StandardScaler(), est_for_cv)

    cv_res = cross_validate(est_for_cv, X_tr, y_tr, cv=cv, scoring={"f1": f1_scorer})
    return float(cv_res["test_f1"].mean()), float(cv_res["test_f1"].std())


def _evaluate_flaml(X_tr, y_tr, X_te, y_te, cv, time_budget, random_state):
    estimator_list = _available_flaml_estimators()
    automl = AutoML()

    t0 = time.perf_counter()
    automl.fit(
        X_tr,
        y_tr,
        task="classification",
        metric="f1",
        time_budget=time_budget,
        eval_method="cv",
        n_splits=5,
        split_type="stratified",
        estimator_list=estimator_list,
        seed=random_state,
        verbose=0,
        log_file_name="",
    )
    train_time_sec = time.perf_counter() - t0

    cv_f1_mean, cv_f1_std = _evaluate_flaml_cv(automl, X_tr, y_tr, cv)
    pred = automl.predict(X_te)
    proba = automl.predict_proba(X_te)[:, 1]
    latency_ms = measure_inference_latency_ms(automl, X_te)

    return {
        "name": "flaml_automl",
        "method": "automl",
        "best_estimator": str(automl.best_estimator),
        "best_config": _to_json_safe(automl.best_config or {}),
        "cv_f1_mean": round(cv_f1_mean, 4),
        "cv_f1_std": round(cv_f1_std, 4),
        "test_f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "train_time_sec": round(float(train_time_sec), 4),
        "inference_latency_ms": round(float(latency_ms), 4),
    }


def _build_summary(results, dataset_rows=None):
    by_name = {r["name"]: r for r in results}
    flaml_res = by_name["flaml_automl"]
    manual_res = by_name["logreg_C10"]
    cgpa_res = by_name["cgpa_only_logreg"]

    best_cv = max(results, key=lambda r: r["cv_f1_mean"])
    delta_vs_manual = round(flaml_res["cv_f1_mean"] - manual_res["cv_f1_mean"], 4)
    within_one_std = abs(delta_vs_manual) <= max(
        flaml_res["cv_f1_std"], manual_res["cv_f1_std"]
    )
    outperforms_manual = delta_vs_manual > 0 and not within_one_std
    row_label = (
        f"{dataset_rows} records" if dataset_rows is not None else "this dataset"
    )

    if outperforms_manual:
        takeaway = (
            f"FLAML ({flaml_res['best_estimator']}) outperformed manual logreg_C10 "
            f"by {delta_vs_manual:+.4f} CV F1 (outside 1 SD) on {row_label}."
        )
    else:
        takeaway = (
            f"On {row_label}, FLAML ({flaml_res['best_estimator']}, CV F1 "
            f"{flaml_res['cv_f1_mean']:.4f} +/- {flaml_res['cv_f1_std']:.4f}) does not "
            f"meaningfully outperform the simple manual champion logreg_C10 (CV F1 "
            f"{manual_res['cv_f1_mean']:.4f} +/- {manual_res['cv_f1_std']:.4f}; "
            f"delta {delta_vs_manual:+.4f}, within 1 SD) or the CGPA-only baseline "
            f"(CV F1 {cgpa_res['cv_f1_mean']:.4f}). Keeping logreg_C10 is preferable "
            f"for simplicity, lower training cost ({manual_res['train_time_sec']:.4f}s "
            f"vs {flaml_res['train_time_sec']:.2f}s), and fast inference."
        )

    return {
        "best_cv_f1_model": best_cv["name"],
        "flaml_vs_manual_cv_f1_delta": delta_vs_manual,
        "within_one_std": within_one_std,
        "automl_outperforms_manual": outperforms_manual,
        "takeaway": takeaway,
    }


def _render_markdown(payload):
    meta = payload["metadata"]
    results = payload["results"]
    summary = payload["summary"]
    flaml_res = next(r for r in results if r["name"] == "flaml_automl")

    lines = [
        "# AutoML (FLAML) vs. Manual Training Comparison",
        "",
        "## Experimental Setup",
        "",
        (
            f"- **Dataset**: `{meta['data_path']}` ({meta['dataset_rows']} rows, "
            f"target: `Chance of Admit >= 0.75`)"
        ),
        (
            f"- **Split**: Stratified 80/20 (`{meta['train_rows']}` train / "
            f"`{meta['test_rows']}` sealed test, `random_state={meta['random_state']}`)"
        ),
        f"- **Cross-Validation**: {meta['cv_protocol']} (evaluated on training split only)",
        (
            f"- **FLAML Time Budget**: `{meta['time_budget_sec']}s` "
            f"(internal 5-fold stratified CV optimizing `f1`; discovered estimator "
            f"re-scored with the exact 5x3 repeated CV protocol)"
        ),
        f"- **MLflow Experiment**: `{meta['mlflow_experiment']}`",
        "",
        "## Benchmark Results",
        "",
        (
            "| Model | Method | Best Estimator | CV F1 (mean +/- std) | Test F1 | "
            "Test ROC-AUC | Train Time (s) | Inference Latency (ms/sample) |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| `{r['name']}` | {r['method']} | `{r['best_estimator']}` | "
            f"{r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f} | "
            f"{r['test_f1']:.4f} | {r['test_roc_auc']:.4f} | "
            f"{r['train_time_sec']:.4f} | {r['inference_latency_ms']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## FLAML Best Configuration",
            "",
            f"- **Best Estimator**: `{flaml_res['best_estimator']}`",
            "- **Hyperparameters**:",
            "```json",
            json.dumps(flaml_res["best_config"], indent=2, sort_keys=True),
            "```",
            "",
            f"## Conclusion: Does AutoML Outperform Simple/Manual Models on {meta['dataset_rows']} Records?",
            "",
            summary["takeaway"],
            "",
        ]
    )
    return "\n".join(lines)


def run_comparison(
    time_budget=DEFAULT_TIME_BUDGET,
    data_path=DEFAULT_DATA_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    tracking_uri=None,
    random_state=DEFAULT_RANDOM_STATE,
):
    df = pd.read_csv(data_path)
    X, y = df[FEATURES].values, df["target"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)

    resolved_uri = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
    )
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = [
        _evaluate_manual_model(
            "logreg_C10",
            "manual_champion",
            "logreg",
            {"C": 10.0},
            X_tr,
            y_tr,
            X_te,
            y_te,
            cv,
        ),
        _evaluate_manual_model(
            "cgpa_only_logreg",
            "manual_baseline",
            "cgpa_only",
            {},
            X_tr,
            y_tr,
            X_te,
            y_te,
            cv,
        ),
        _evaluate_flaml(X_tr, y_tr, X_te, y_te, cv, time_budget, random_state),
    ]

    for r in results:
        with mlflow.start_run(run_name=r["name"]):
            mlflow.log_params(
                {
                    "method": r["method"],
                    "best_estimator": r["best_estimator"],
                    "best_config": json.dumps(r["best_config"], sort_keys=True)[:500],
                    "cv_splits": 5,
                    "cv_repeats": 3,
                    "random_state": random_state,
                    "time_budget_sec": time_budget if r["method"] == "automl" else 0,
                }
            )
            mlflow.log_metrics(
                {
                    "cv_f1_mean": r["cv_f1_mean"],
                    "cv_f1_std": r["cv_f1_std"],
                    "test_f1": r["test_f1"],
                    "test_roc_auc": r["test_roc_auc"],
                    "train_time_sec": r["train_time_sec"],
                    "inference_latency_ms": r["inference_latency_ms"],
                }
            )

    payload = {
        "metadata": {
            "data_path": str(data_path),
            "dataset_rows": len(df),
            "train_rows": len(X_tr),
            "test_rows": len(X_te),
            "cv_protocol": "RepeatedStratifiedKFold(n_splits=5, n_repeats=3)",
            "time_budget_sec": float(time_budget),
            "random_state": int(random_state),
            "mlflow_experiment": EXPERIMENT_NAME,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        "results": results,
        "summary": _build_summary(results, dataset_rows=len(df)),
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "automl_comparison.json"
    md_path = out_path / "automl_comparison.md"

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare bounded FLAML AutoML against manual reference models."
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=DEFAULT_TIME_BUDGET,
        help="FLAML search time budget in seconds (default: 20).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to processed dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write comparison reports.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Optional MLflow tracking URI (defaults to MLFLOW_TRACKING_URI or sqlite:///mlflow.db).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for split, CV, and FLAML (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    payload = run_comparison(
        time_budget=args.time_budget,
        data_path=args.data_path,
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
        random_state=args.random_state,
    )
    for r in payload["results"]:
        print(
            f"{r['name']:18s} est={r['best_estimator']:10s} "
            f"cv_f1={r['cv_f1_mean']:.4f} (+/-{r['cv_f1_std']:.4f}) "
            f"test_f1={r['test_f1']:.4f} test_auc={r['test_roc_auc']:.4f} "
            f"train={r['train_time_sec']:.4f}s lat={r['inference_latency_ms']:.4f}ms"
        )
    print("\n" + payload["summary"]["takeaway"])


if __name__ == "__main__":
    main()
