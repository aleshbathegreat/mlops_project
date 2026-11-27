"""Manual ML vs AutoML (FLAML) on the same data, over several seeds.

Research question: can AutoML reach comparable or better predictive performance
than manual machine learning, while reducing development time and effort?

For every seed the data is split 80/20 (stratified). Each approach then
develops a model on the training split only and is scored once on the sealed
test split:

* manual  - the full manual search from train.py (every non-baseline model,
            repeated CV on ROC-AUC, simplest model within one SD of the best)
* automl  - FLAML with a fixed time budget, for several budgets

Development time is the wall-clock time to go from the training data to a
fitted, selected model (search plus final fit).
"""

import argparse
import inspect
import json
import os
import pickle
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.utils.class_weight import compute_sample_weight

try:
    from src.automl_compare import (
        _available_flaml_estimators,
        measure_inference_latency_ms,
    )
    from src.train import (
        BASELINES,
        CONFIGS,
        CV_REPEATS,
        CV_SPLITS,
        FEATURES,
        build,
        pick_champion,
    )
except ModuleNotFoundError:
    from automl_compare import _available_flaml_estimators, measure_inference_latency_ms
    from train import (
        BASELINES,
        CONFIGS,
        CV_REPEATS,
        CV_SPLITS,
        FEATURES,
        build,
        pick_champion,
    )

warnings.filterwarnings("ignore")

DEFAULT_DATA_PATH = "data/processed.csv"
DEFAULT_OUTPUT_DIR = "reports"
DEFAULT_N_SEEDS = 5
DEFAULT_BASE_SEED = 42
DEFAULT_BUDGETS = (15.0, 60.0)
EXPERIMENT_NAME = "manual-vs-automl-3k"
METRICS = [
    "test_auc",
    "test_bal_acc",
    "test_f1",
    "dev_time_sec",
    "latency_ms",
    "model_size_kb",
]


def _budget_label(budget):
    return f"automl_{budget:g}s"


def _score(model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    try:
        size_kb = len(pickle.dumps(model)) / 1024.0
    except Exception:  # noqa: BLE001 - size is informational only
        size_kb = float("nan")
    return {
        "test_auc": float(roc_auc_score(y_te, proba)),
        "test_bal_acc": float(balanced_accuracy_score(y_te, pred)),
        "test_f1": float(f1_score(y_te, pred, zero_division=0)),
        "latency_ms": measure_inference_latency_ms(model, X_te),
        "model_size_kb": float(size_kb),
    }


def run_manual(X_tr, y_tr, X_te, y_te, seed, config_names=None):
    configs = [
        c
        for c in CONFIGS
        if c[1] not in BASELINES and (config_names is None or c[0] in config_names)
    ]
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=seed
    )

    t0 = time.perf_counter()
    rows = []
    for name, family, params in configs:
        scores = cross_val_score(
            build(family, params), X_tr, y_tr, cv=cv, scoring="roc_auc"
        )
        rows.append(
            (name, family, float(scores.mean()), float(scores.std()), 0.0, 0.0, 0.0)
        )
    champion = pick_champion(rows)
    family, params = next((f, p) for n, f, p in CONFIGS if n == champion[0])
    model = build(family, params)
    model.fit(X_tr, y_tr)
    dev_time = time.perf_counter() - t0

    out = _score(model, X_te, y_te)
    out.update(
        approach="manual",
        seed=seed,
        chosen=champion[0],
        cv_auc=champion[2],
        dev_time_sec=dev_time,
        models_tried=len(configs),
    )
    return out


def _weighted_estimators():
    """FLAML estimators that accept class weights (its KNN does not)."""
    return [e for e in _available_flaml_estimators() if e != "kneighbor"]


def run_automl(X_tr, y_tr, X_te, y_te, seed, budget):
    automl = AutoML()
    weights = compute_sample_weight("balanced", y_tr)  # same class weighting as manual
    t0 = time.perf_counter()
    automl.fit(
        X_tr,
        y_tr,
        sample_weight=weights,
        task="classification",
        metric="roc_auc",
        time_budget=budget,
        eval_method="cv",
        n_splits=5,
        split_type="stratified",
        estimator_list=_weighted_estimators(),
        seed=seed,
        verbose=0,
        log_file_name="",
    )
    dev_time = time.perf_counter() - t0

    out = _score(automl, X_te, y_te)
    out.update(
        approach=_budget_label(budget),
        seed=seed,
        chosen=str(automl.best_estimator),
        cv_auc=float(1.0 - automl.best_loss),
        dev_time_sec=dev_time,
        models_tried=None,
    )
    return out


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std}


def aggregate(runs):
    approaches = ["manual"] + sorted(
        {r["approach"] for r in runs if r["approach"] != "manual"},
        key=lambda a: float(a.split("_")[1].rstrip("s")),
    )
    summary = {}
    for a in approaches:
        rs = [r for r in runs if r["approach"] == a]
        entry = {"n_seeds": len(rs)}
        for m in METRICS:
            entry[m] = _mean_std([r[m] for r in rs])
        chosen = {}
        for r in rs:
            chosen[r["chosen"]] = chosen.get(r["chosen"], 0) + 1
        entry["chosen"] = chosen
        summary[a] = entry

    manual_by_seed = {r["seed"]: r for r in runs if r["approach"] == "manual"}
    paired = {}
    for a in approaches[1:]:
        diffs = [
            r["test_auc"] - manual_by_seed[r["seed"]]["test_auc"]
            for r in runs
            if r["approach"] == a
        ]
        stats = _mean_std(diffs)
        if stats["mean"] - stats["std"] > 0:
            verdict = "better"
        elif stats["mean"] + stats["std"] < 0:
            verdict = "worse"
        else:
            verdict = "comparable"
        speedup = summary["manual"]["dev_time_sec"]["mean"] / max(
            summary[a]["dev_time_sec"]["mean"], 1e-9
        )
        paired[a] = {
            "auc_diff_mean": stats["mean"],
            "auc_diff_std": stats["std"],
            "wins": int(sum(d > 0 for d in diffs)),
            "n": len(diffs),
            "verdict": verdict,
            "dev_time_ratio_manual_over_automl": float(speedup),
        }
    return summary, paired


def effort_proxy():
    """Lines of code written for each approach (a rough proxy for effort)."""

    def count(src):
        return sum(
            1
            for line in src.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    train_path = Path(inspect.getsourcefile(build)).with_name("train.py")
    manual = count(train_path.read_text(encoding="utf-8"))
    auto = count(inspect.getsource(run_automl))
    return {"manual_loc": manual, "automl_loc": auto}


def build_conclusion(summary, paired, n_rows):
    man = summary["manual"]
    parts = []
    for a, p in paired.items():
        s = summary[a]
        parts.append(
            f"With a {a.split('_')[1]} budget AutoML was {p['verdict']} "
            f"(test AUC {s['test_auc']['mean']:.4f} vs {man['test_auc']['mean']:.4f}; "
            f"paired difference {p['auc_diff_mean']:+.4f} +/- {p['auc_diff_std']:.4f}, "
            f"AutoML ahead in {p['wins']}/{p['n']} seeds) and finished "
            f"{p['dev_time_ratio_manual_over_automl']:.1f}x "
            f"{'faster' if p['dev_time_ratio_manual_over_automl'] >= 1 else 'slower'} "
            f"than the manual search ({s['dev_time_sec']['mean']:.0f}s vs "
            f"{man['dev_time_sec']['mean']:.0f}s)."
        )
    return f"On {n_rows} records over {man['n_seeds']} seeds. " + " ".join(parts)


def _fmt(x, d=4):
    return f"{x['mean']:.{d}f} +/- {x['std']:.{d}f}"


def render_markdown(payload):
    meta, summary, paired = payload["metadata"], payload["summary"], payload["paired"]
    lines = [
        "# Manual ML vs AutoML (FLAML)",
        "",
        (
            "**Research question:** can AutoML achieve comparable or better "
            "predictive performance than manual machine learning, while reducing "
            "model development time and effort?"
        ),
        "",
        "## Setup",
        "",
        f"- Dataset: `{meta['data_path']}` ({meta['dataset_rows']} rows, about 79% admitted)",
        f"- Seeds (independent 80/20 stratified splits): {meta['seeds']}",
        (
            f"- Manual: {meta['manual_models_tried']} model configurations, "
            f"{meta['cv_protocol']}, champion = best CV ROC-AUC then simplest "
            "within 1 SD"
        ),
        (
            f"- AutoML: FLAML, ROC-AUC objective, time budgets "
            f"{meta['budgets_sec']} s, same balanced class weights as the manual "
            "models"
        ),
        (
            "- Every model is developed on the training split only and scored "
            "once on the sealed test split"
        ),
        f"- MLflow experiment: `{meta['mlflow_experiment']}`",
        "",
        "## Performance and reliability (mean +/- std over seeds)",
        "",
        "| Approach | Test ROC-AUC | Test balanced acc | Test F1 |",
        "|---|---|---|---|",
    ]
    for a, s in summary.items():
        lines.append(
            f"| `{a}` | {_fmt(s['test_auc'])} | {_fmt(s['test_bal_acc'])} | "
            f"{_fmt(s['test_f1'])} |"
        )
    lines += [
        "",
        "## AutoML vs manual, paired by seed (test ROC-AUC)",
        "",
        "| AutoML budget | Mean difference | Seeds AutoML won | Verdict |",
        "|---|---|---|---|",
    ]
    for a, p in paired.items():
        lines.append(
            f"| `{a}` | {p['auc_diff_mean']:+.4f} +/- {p['auc_diff_std']:.4f} | "
            f"{p['wins']}/{p['n']} | {p['verdict']} |"
        )
    lines += [
        "",
        (
            "Verdict rule: *better* if the mean difference is more than one SD "
            "above zero, *worse* if more than one SD below, otherwise *comparable*."
        ),
        "",
        "## Efficiency",
        "",
        "| Approach | Development time (s) | Inference (ms/row) | Model size (KB) |",
        "|---|---|---|---|",
    ]
    for a, s in summary.items():
        lines.append(
            f"| `{a}` | {_fmt(s['dev_time_sec'], 1)} | "
            f"{_fmt(s['latency_ms'], 3)} | {_fmt(s['model_size_kb'], 0)} |"
        )
    eff = payload["effort"]
    lines += [
        "",
        (
            f"Effort proxy (lines of code): manual pipeline {eff['manual_loc']}, "
            f"AutoML call {eff['automl_loc']}. Lines of code ignore the human "
            "time spent choosing models and ranges; record your own hours in "
            "the README."
        ),
        "",
        "## Which model each approach picked",
        "",
    ]
    for a, s in summary.items():
        picks = ", ".join(f"{k} x{v}" for k, v in sorted(s["chosen"].items()))
        lines.append(f"- `{a}`: {picks}")
    lines += ["", "## Conclusion", "", payload["conclusion"], ""]
    return "\n".join(lines)


def run_experiment(
    n_seeds=DEFAULT_N_SEEDS,
    budgets=DEFAULT_BUDGETS,
    config_names=None,
    data_path=DEFAULT_DATA_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    tracking_uri=None,
    base_seed=DEFAULT_BASE_SEED,
):
    df = pd.read_csv(data_path)
    X, y = df[FEATURES].values, df["target"].values
    seeds = [base_seed + i for i in range(n_seeds)]

    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    runs = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        for job in ["manual", *budgets]:
            if job == "manual":
                r = run_manual(X_tr, y_tr, X_te, y_te, seed, config_names)
            else:
                r = run_automl(X_tr, y_tr, X_te, y_te, seed, job)
            label = r["approach"]
            runs.append(r)
            with mlflow.start_run(run_name=f"{label}_seed{seed}"):
                mlflow.log_params(
                    {"approach": label, "seed": seed, "chosen": r["chosen"]}
                )
                mlflow.log_metrics({m: r[m] for m in METRICS})
            print(
                f"seed={seed} {label:12s} chosen={r['chosen']:14s} "
                f"test_auc={r['test_auc']:.4f} dev_time={r['dev_time_sec']:.1f}s",
                flush=True,
            )

    summary, paired = aggregate(runs)
    manual_tried = next(r["models_tried"] for r in runs if r["approach"] == "manual")
    payload = {
        "metadata": {
            "data_path": str(data_path),
            "dataset_rows": len(df),
            "seeds": seeds,
            "budgets_sec": [float(b) for b in budgets],
            "manual_models_tried": manual_tried,
            "cv_protocol": (
                f"RepeatedStratifiedKFold(n_splits={CV_SPLITS}, n_repeats={CV_REPEATS})"
            ),
            "mlflow_experiment": EXPERIMENT_NAME,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        "runs": runs,
        "summary": summary,
        "paired": paired,
        "effort": effort_proxy(),
    }
    payload["conclusion"] = build_conclusion(summary, paired, len(df))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manual_vs_automl.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out / "manual_vs_automl.md").write_text(render_markdown(payload), encoding="utf-8")
    return payload


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Manual ML vs AutoML over several seeds.")
    p.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS, help="number of seeds")
    p.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="FLAML time budgets in seconds",
    )
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    payload = run_experiment(
        n_seeds=a.seeds,
        budgets=a.budgets,
        data_path=a.data_path,
        output_dir=a.output_dir,
        tracking_uri=a.tracking_uri,
        base_seed=a.base_seed,
    )
    print("\n" + payload["conclusion"])


if __name__ == "__main__":
    main()
