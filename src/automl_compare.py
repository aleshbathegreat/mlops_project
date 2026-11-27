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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.train import CONFIGS, CV_REPEATS, CV_SPLITS, FEATURES, build
except ModuleNotFoundError:
    from train import CONFIGS, CV_REPEATS, CV_SPLITS, FEATURES, build

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = "data/processed.csv"
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_TIME_BUDGET = 60.0
DEFAULT_RANDOM_STATE = 42
EXPERIMENT_NAME = "admission-automl-comparison-3k"

# Manual models taken from train.py so the names always match the MLflow runs.
MANUAL_CHAMPION = "svm_rbf_C10"
SIMPLE_MODEL = "logreg_C10"
BASELINE = "cgpa_only_logreg"
CV_SCORING = {"auc": "roc_auc", "bal_acc": "balanced_accuracy"}


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


def measure_inference_latency_ms(model, X_te, max_rows=200):
    rows = X_te[:max_rows]
    start = time.perf_counter()
    for row in rows:
        sample = row.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            model.predict_proba(sample)
        else:
            model.predict(sample)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return float(elapsed_ms / len(rows))


def _summarise(name, method, estimator, config, cv_res, model, fit_time, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    return {
        "name": name,
        "method": method,
        "best_estimator": estimator,
        "best_config": _to_json_safe(config),
        "cv_auc_mean": round(float(cv_res["test_auc"].mean()), 4),
        "cv_auc_std": round(float(cv_res["test_auc"].std()), 4),
        "cv_bal_acc_mean": round(float(cv_res["test_bal_acc"].mean()), 4),
        "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "test_bal_acc": round(float(balanced_accuracy_score(y_te, pred)), 4),
        "test_f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "train_time_sec": round(float(fit_time), 4),
        "inference_latency_ms": round(measure_inference_latency_ms(model, X_te), 4),
    }


def _evaluate_manual_model(name, method, X_tr, y_tr, X_te, y_te, cv):
    family, params = {n: (f, p) for n, f, p in CONFIGS}[name]
    model = build(family, params)
    cv_res = cross_validate(model, X_tr, y_tr, cv=cv, scoring=CV_SCORING)

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    return _summarise(name, method, family, params, cv_res, model, fit_time, X_te, y_te)


def _evaluate_flaml(X_tr, y_tr, X_te, y_te, cv, time_budget, random_state):
    automl = AutoML()
    t0 = time.perf_counter()
    automl.fit(
        X_tr,
        y_tr,
        task="classification",
        metric="roc_auc",
        time_budget=time_budget,
        eval_method="cv",
        n_splits=5,
        split_type="stratified",
        estimator_list=_available_flaml_estimators(),
        seed=random_state,
        verbose=0,
        log_file_name="",
    )
    fit_time = time.perf_counter() - t0

    # Re-score the winning estimator with the same repeated CV as the manual models.
    estimator = automl.model.estimator
    try:
        est_for_cv = clone(estimator)
    except (TypeError, ValueError):
        est_for_cv = estimator
    if str(automl.best_estimator) in {"lrl1", "lrl2", "kneighbor"}:
        est_for_cv = make_pipeline(StandardScaler(), est_for_cv)
    cv_res = cross_validate(est_for_cv, X_tr, y_tr, cv=cv, scoring=CV_SCORING)

    return _summarise(
        "flaml_automl",
        "automl",
        str(automl.best_estimator),
        automl.best_config or {},
        cv_res,
        automl,
        fit_time,
        X_te,
        y_te,
    )


def _build_summary(results, n_rows):
    by_name = {r["name"]: r for r in results}
    flaml_res = by_name["flaml_automl"]
    champ = by_name[MANUAL_CHAMPION]
    simple = by_name[SIMPLE_MODEL]
    base = by_name[BASELINE]

    best_cv = max(results, key=lambda r: r["cv_auc_mean"])
    delta = round(flaml_res["cv_auc_mean"] - champ["cv_auc_mean"], 4)
    within_one_std = abs(delta) <= max(flaml_res["cv_auc_std"], champ["cv_auc_std"])
    outperforms = delta > 0 and not within_one_std

    flaml_txt = (
        f"FLAML ({flaml_res['best_estimator']}, CV ROC-AUC "
        f"{flaml_res['cv_auc_mean']:.4f} +/- {flaml_res['cv_auc_std']:.4f})"
    )
    champ_txt = (
        f"the manual champion {MANUAL_CHAMPION} (CV ROC-AUC "
        f"{champ['cv_auc_mean']:.4f} +/- {champ['cv_auc_std']:.4f})"
    )
    if outperforms:
        takeaway = (
            f"On {n_rows} records, {flaml_txt} outperformed {champ_txt} by "
            f"{delta:+.4f} (outside 1 SD). Fit times cover the final fit only "
            f"({flaml_res['train_time_sec']:.2f}s vs "
            f"{champ['train_time_sec']:.2f}s); the manual champion was chosen "
            "after a longer model search."
        )
    else:
        takeaway = (
            f"On {n_rows} records, {flaml_txt} does not meaningfully outperform "
            f"{champ_txt} (delta {delta:+.4f}, within 1 SD). Both clearly beat the "
            f"simple {SIMPLE_MODEL} ({simple['cv_auc_mean']:.4f}) and the CGPA-only "
            f"baseline ({base['cv_auc_mean']:.4f}). Fit times cover the final fit "
            f"only ({champ['train_time_sec']:.2f}s manual vs "
            f"{flaml_res['train_time_sec']:.2f}s AutoML); the manual champion was "
            "chosen after a longer model search, so the development-time "
            "comparison is in reports/manual_vs_automl.md."
        )

    return {
        "best_cv_auc_model": best_cv["name"],
        "flaml_vs_champion_cv_auc_delta": delta,
        "within_one_std": within_one_std,
        "automl_outperforms_manual": outperforms,
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
            "target: `Chance of Admit >= 0.75`, about 79% admitted)"
        ),
        (
            f"- **Split**: Stratified 80/20 (`{meta['train_rows']}` train / "
            f"`{meta['test_rows']}` sealed test, `random_state={meta['random_state']}`)"
        ),
        f"- **Cross-Validation**: {meta['cv_protocol']} (training split only)",
        (
            f"- **FLAML Time Budget**: `{meta['time_budget_sec']}s` "
            "(internal 5-fold stratified CV optimizing `roc_auc`; the discovered "
            "estimator is re-scored with the same repeated CV as the manual models)"
        ),
        f"- **MLflow Experiment**: `{meta['mlflow_experiment']}`",
        "",
        "## Benchmark Results",
        "",
        (
            "| Model | Method | Best Estimator | CV ROC-AUC (mean +/- std) | "
            "CV Balanced Acc | Test ROC-AUC | Test Balanced Acc | "
            "Fit Time (s) | Latency (ms/sample) |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['name']}` | {r['method']} | `{r['best_estimator']}` | "
            f"{r['cv_auc_mean']:.4f} +/- {r['cv_auc_std']:.4f} | "
            f"{r['cv_bal_acc_mean']:.4f} | {r['test_roc_auc']:.4f} | "
            f"{r['test_bal_acc']:.4f} | {r['train_time_sec']:.2f} | "
            f"{r['inference_latency_ms']:.4f} |"
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
            f"## Conclusion: Does AutoML Beat Manual Models on {meta['dataset_rows']} Records?",
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
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=random_state
    )

    resolved_uri = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
    )
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = [
        _evaluate_manual_model(
            MANUAL_CHAMPION, "manual_champion", X_tr, y_tr, X_te, y_te, cv
        ),
        _evaluate_manual_model(
            SIMPLE_MODEL, "manual_simple", X_tr, y_tr, X_te, y_te, cv
        ),
        _evaluate_manual_model(BASELINE, "manual_baseline", X_tr, y_tr, X_te, y_te, cv),
        _evaluate_flaml(X_tr, y_tr, X_te, y_te, cv, time_budget, random_state),
    ]

    for r in results:
        with mlflow.start_run(run_name=r["name"]):
            mlflow.log_params(
                {
                    "method": r["method"],
                    "best_estimator": r["best_estimator"],
                    "best_config": json.dumps(r["best_config"], sort_keys=True)[:500],
                    "cv_splits": CV_SPLITS,
                    "cv_repeats": CV_REPEATS,
                    "random_state": random_state,
                    "time_budget_sec": time_budget if r["method"] == "automl" else 0,
                }
            )
            mlflow.log_metrics(
                {
                    "cv_auc_mean": r["cv_auc_mean"],
                    "cv_auc_std": r["cv_auc_std"],
                    "cv_bal_acc_mean": r["cv_bal_acc_mean"],
                    "test_roc_auc": r["test_roc_auc"],
                    "test_bal_acc": r["test_bal_acc"],
                    "test_f1": r["test_f1"],
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
            "cv_protocol": (
                f"RepeatedStratifiedKFold(n_splits={CV_SPLITS}, n_repeats={CV_REPEATS})"
            ),
            "time_budget_sec": float(time_budget),
            "random_state": int(random_state),
            "mlflow_experiment": EXPERIMENT_NAME,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        "results": results,
        "summary": _build_summary(results, len(df)),
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "automl_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_path / "automl_comparison.md").write_text(
        _render_markdown(payload), encoding="utf-8"
    )
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare bounded FLAML AutoML against manual reference models."
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=DEFAULT_TIME_BUDGET,
        help="FLAML search time budget in seconds (default: 60).",
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
            f"cv_auc={r['cv_auc_mean']:.4f} (+/-{r['cv_auc_std']:.4f}) "
            f"test_auc={r['test_roc_auc']:.4f} test_bal_acc={r['test_bal_acc']:.4f} "
            f"fit={r['train_time_sec']:.2f}s lat={r['inference_latency_ms']:.3f}ms"
        )
    print("\n" + payload["summary"]["takeaway"])


if __name__ == "__main__":
    main()
