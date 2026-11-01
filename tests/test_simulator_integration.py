from pathlib import Path
import pandas as pd
import json


def test_drift_metrics_csv_structure_and_values(synthetic_simulation_artifacts):
    drift_csv, _ = synthetic_simulation_artifacts
    assert drift_csv.exists(), "drift_metrics.csv should be created by the simulator"

    df = pd.read_csv(drift_csv)
    assert len(df) == 8, "Expected 8 weekly simulation records"

    expected_cols = {
        "week",
        "n_labeled",
        "psi_mean",
        "psi_max",
        "psi_status",
        "never_retrain_auc_reference",
        "retrain_model_auc_recent",
        "retrained",
        "trigger_reason",
    }
    assert expected_cols.issubset(set(df.columns))

    # Pre-drift weeks (1-4) should be stable
    pre_drift = df[df["week"] <= 4]
    assert (pre_drift["psi_status"] == "Stable").all()

    # Drifted weeks (5-8) should detect significant shift
    post_drift = df[df["week"] >= 5]
    assert (post_drift["psi_status"] == "Significant Shift (Drift Detected)").all()
    assert (post_drift["psi_max"] >= 0.25).all()

    # Verify retraining triggered at least in week 3 (initial) and week 7 (decay)
    retrained_weeks = df[df["retrained"] == True]["week"].tolist()
    assert 3 in retrained_weeks
    assert 7 in retrained_weeks


def test_retraining_artifacts_saved(synthetic_simulation_artifacts):
    _, retrain_dir = synthetic_simulation_artifacts
    assert retrain_dir.exists()

    # Check week 7 retraining artifacts
    week7_dir = retrain_dir / "week_7"
    assert week7_dir.exists()
    assert (week7_dir / "metrics.json").exists()
    assert (week7_dir / "model.joblib").exists()

    with open(week7_dir / "metrics.json") as f:
        metrics = json.load(f)

    assert metrics["week"] == 7
    assert "performance decay" in metrics["trigger"]
    assert metrics["sample_decay_rate"] == 0.85
    assert metrics["auc_on_recent_live"] > 0.80
