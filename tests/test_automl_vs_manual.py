import json

import pandas as pd

from src.automl_vs_manual import aggregate, parse_args, run_experiment


def test_parse_args_defaults_and_override():
    args = parse_args([])
    assert args.seeds == 5
    assert args.budgets == [15.0, 60.0]

    custom = parse_args(["--seeds", "2", "--budgets", "3", "9"])
    assert custom.seeds == 2
    assert custom.budgets == [3.0, 9.0]


def test_aggregate_paired_verdicts():
    def run(approach, seed, auc, t):
        return {
            "approach": approach,
            "seed": seed,
            "chosen": "x",
            "test_auc": auc,
            "test_bal_acc": 0.9,
            "test_f1": 0.9,
            "dev_time_sec": t,
            "latency_ms": 0.1,
            "model_size_kb": 10.0,
        }

    runs = []
    for seed, base in enumerate([0.90, 0.91, 0.92]):
        runs.append(run("manual", seed, base, 100.0))
        runs.append(run("automl_15s", seed, base + 0.05, 15.0))  # clearly better
        runs.append(run("automl_60s", seed, base - 0.05, 60.0))  # clearly worse
    summary, paired = aggregate(runs)

    assert list(summary) == ["manual", "automl_15s", "automl_60s"]
    assert paired["automl_15s"]["verdict"] == "better"
    assert paired["automl_15s"]["wins"] == 3
    assert paired["automl_60s"]["verdict"] == "worse"
    assert paired["automl_15s"]["dev_time_ratio_manual_over_automl"] > 6


def test_run_experiment_small(tmp_path):
    payload = run_experiment(
        n_seeds=1,
        budgets=[2],
        config_names={"logreg_C1", "rf_100_depth3", "svm_rbf_C1"},
        output_dir=str(tmp_path),
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )

    n_rows = len(pd.read_csv("data/processed.csv"))
    saved = json.loads((tmp_path / "manual_vs_automl.json").read_text("utf-8"))
    assert saved["metadata"]["dataset_rows"] == n_rows
    assert saved["metadata"]["manual_models_tried"] == 3
    assert {r["approach"] for r in saved["runs"]} == {"manual", "automl_2s"}

    for r in saved["runs"]:
        assert 0.5 < r["test_auc"] <= 1.0
        assert 0.0 <= r["test_bal_acc"] <= 1.0
        assert r["dev_time_sec"] > 0.0
        assert r["latency_ms"] > 0.0

    assert saved["paired"]["automl_2s"]["verdict"] in {"better", "worse", "comparable"}
    assert saved["effort"]["manual_loc"] > saved["effort"]["automl_loc"] > 0

    md = (tmp_path / "manual_vs_automl.md").read_text("utf-8")
    assert "## Conclusion" in md
    assert payload["conclusion"] in md
