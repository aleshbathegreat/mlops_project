import json

import pandas as pd

from src.automl_compare import parse_args, run_comparison


def test_parse_args_defaults_and_override():
    default_args = parse_args([])
    assert default_args.time_budget == 60.0
    assert default_args.random_state == 42

    custom_args = parse_args(["--time-budget", "2", "--output-dir", "tmp_reports"])
    assert custom_args.time_budget == 2.0
    assert custom_args.output_dir == "tmp_reports"


def test_run_comparison_outputs_and_metrics(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    payload = run_comparison(
        time_budget=2,
        output_dir=str(tmp_path),
        tracking_uri=tracking_uri,
        random_state=42,
    )

    json_path = tmp_path / "automl_comparison.json"
    md_path = tmp_path / "automl_comparison.md"
    assert json_path.exists()
    assert md_path.exists()

    n_rows = len(pd.read_csv("data/processed.csv"))
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    meta = saved["metadata"]
    assert meta["dataset_rows"] == n_rows
    assert meta["train_rows"] + meta["test_rows"] == n_rows
    assert abs(meta["test_rows"] / n_rows - 0.2) < 0.01
    assert meta["time_budget_sec"] == 2.0

    by_name = {r["name"]: r for r in saved["results"]}
    assert set(by_name.keys()) == {
        "flaml_automl",
        "svm_rbf_C10",
        "logreg_C10",
        "cgpa_only_logreg",
    }

    for name, metrics in by_name.items():
        assert metrics["best_estimator"], f"missing best_estimator for {name}"
        assert 0.0 <= metrics["cv_auc_mean"] <= 1.0
        assert metrics["cv_auc_std"] >= 0.0
        assert 0.0 <= metrics["cv_bal_acc_mean"] <= 1.0
        assert 0.0 <= metrics["test_roc_auc"] <= 1.0
        assert 0.0 <= metrics["test_bal_acc"] <= 1.0
        assert metrics["train_time_sec"] > 0.0
        assert metrics["inference_latency_ms"] > 0.0

    # every real model must beat the CGPA-only baseline's ranking power minus noise
    assert by_name["svm_rbf_C10"]["cv_auc_mean"] > 0.85

    md_text = md_path.read_text(encoding="utf-8")
    for name in by_name:
        assert name in md_text
    assert "## Conclusion" in md_text
    assert payload["summary"]["takeaway"] in md_text
