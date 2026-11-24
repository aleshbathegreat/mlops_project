import json

from src.automl_compare import parse_args, run_comparison


def test_parse_args_defaults_and_override():
    default_args = parse_args([])
    assert default_args.time_budget == 20.0
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

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["metadata"]["dataset_rows"] == 400
    assert saved["metadata"]["train_rows"] == 320
    assert saved["metadata"]["test_rows"] == 80
    assert saved["metadata"]["time_budget_sec"] == 2.0

    by_name = {r["name"]: r for r in saved["results"]}
    assert set(by_name.keys()) == {"flaml_automl", "logreg_C10", "cgpa_only_logreg"}

    for name, metrics in by_name.items():
        assert metrics["best_estimator"], f"missing best_estimator for {name}"
        assert 0.0 <= metrics["cv_f1_mean"] <= 1.0
        assert metrics["cv_f1_std"] >= 0.0
        assert 0.0 <= metrics["test_f1"] <= 1.0
        assert 0.0 <= metrics["test_roc_auc"] <= 1.0
        assert metrics["train_time_sec"] > 0.0
        assert metrics["inference_latency_ms"] > 0.0

    md_text = md_path.read_text(encoding="utf-8")
    assert "flaml_automl" in md_text
    assert "logreg_C10" in md_text
    assert "cgpa_only_logreg" in md_text
    assert "## Conclusion" in md_text
    assert payload["summary"]["takeaway"] in md_text
