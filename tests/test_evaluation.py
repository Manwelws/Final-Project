import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

from final_project.evaluation import (
    compute_metrics_at_threshold,
    optimize_threshold,
    run_evaluation,
    ThresholdMetrics,
    EvaluationSummary,
    EvaluationReport,
)


def test_compute_metrics_at_threshold():
    # 4 samples: 2 positive, 2 negative
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.4, 0.6, 0.8])

    # Threshold 0.5 -> predictions [0, 0, 1, 1]
    # TP=2, FP=0, TN=2, FN=0
    metrics = compute_metrics_at_threshold(y_true, y_probs, threshold=0.5, cost_fn=200.0, cost_fp=20.0)
    assert isinstance(metrics, ThresholdMetrics)
    assert metrics.tp == 2
    assert metrics.fp == 0
    assert metrics.tn == 2
    assert metrics.fn == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.f2 == 1.0
    assert metrics.cost == 0.0


def test_optimize_threshold_shifts_lower_under_high_fn_cost():
    # Churners missed (FN) cost 200, False alarms (FP) cost 20
    # True positives have lower probabilities (e.g. 0.35, 0.45)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])

    opt = optimize_threshold(y_true, y_probs, cost_fn=200.0, cost_fp=20.0)
    # Default 0.5 threshold predicts all 0s -> FN=3 -> cost = 3 * 200 = 600
    # Optimal threshold should be <= 0.35 to capture positive churners and reduce total cost
    assert opt.cost_optimal_threshold < 0.5
    assert opt.cost_at_optimal < opt.cost_at_default


def test_run_evaluation_mock_creates_report_and_plots(tmp_path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()

    # Generate synthetic features and target
    X_mat, y_vec = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        weights=[0.75, 0.25],
        random_state=42,
    )
    feature_cols = [f"f_{i}" for i in range(5)]
    pd.DataFrame(X_mat, columns=feature_cols).to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    # Train and save a quick model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_mat, y_vec)
    joblib.dump(clf, models_dir / "model.joblib")
    initial_meta = {
        "algorithm": "RandomForestClassifier",
        "features": feature_cols,
        "cv_auc_mean": 0.85,
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(initial_meta, f)

    summary = run_evaluation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        metadata_path=models_dir / "metadata.json",
        report_dir=reports_dir,
        cost_fn=200.0,
        cost_fp=20.0,
        random_state=42,
    )

    assert isinstance(summary, EvaluationSummary)
    assert 0.5 <= summary.roc_auc <= 1.0
    assert 0.0 <= summary.brier_score <= 1.0

    # Verify report generated
    report_file = reports_dir / "evaluation_report.md"
    assert report_file.exists()
    content = report_file.read_text()
    assert "Holdout Evaluation Report" in content
    assert "Cost-Utility Threshold Optimization" in content

    # Verify figures generated
    figures_dir = reports_dir / "figures"
    assert (figures_dir / "roc_curve.png").exists()
    assert (figures_dir / "pr_curve.png").exists()
    assert (figures_dir / "calibration_curve.png").exists()
    assert (figures_dir / "cost_curve.png").exists()
    assert (figures_dir / "confusion_matrix.png").exists()

    # Verify metadata.json was updated with threshold info
    with open(models_dir / "metadata.json") as f:
        updated_meta = json.load(f)
    assert "optimal_threshold" in updated_meta
    assert "cost_savings" in updated_meta


def test_threshold_metrics_fpr():
    metrics = compute_metrics_at_threshold(
        y_true=np.array([0, 0, 1, 1]),
        y_probs=np.array([0.8, 0.2, 0.9, 0.1]),
        threshold=0.5,
    )
    # y_pred = [1, 0, 1, 0] -> TP=1, FP=1, TN=1, FN=1
    # total negatives = TN + FP = 2 -> FPR = 1/2 = 0.5
    assert metrics.tp == 1
    assert metrics.fp == 1
    assert metrics.tn == 1
    assert metrics.fn == 1
    assert metrics.fpr == 0.5


def test_evaluation_report_in_process_and_rendering(tmp_path):
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6])

    # 1. Pure in-process computation: zero disk writes
    report = EvaluationReport.compute(y_true=y_true, y_probs=y_probs)
    assert isinstance(report, EvaluationReport)
    assert report.roc_auc > 0.9
    assert report.pr_auc > 0.8
    assert 0.0 <= report.brier_score <= 1.0
    assert report.optimization.cost_optimal_threshold <= 0.6

    # 2. Metadata dictionary serialization
    meta_dict = report.to_metadata_dict()
    assert "optimal_threshold" in meta_dict
    assert "cost_savings" in meta_dict
    assert "threshold_comparison" in meta_dict

    # 3. Artifact rendering through report methods
    fig_dir = tmp_path / "figures"
    plots = report.render_plots(fig_dir)
    assert len(plots) == 5
    for p in plots:
        assert Path(p).exists()

    report_path = tmp_path / "report.md"
    report.write_markdown(report_path)
    assert report_path.exists()
    assert "Holdout Evaluation Report" in report_path.read_text()
