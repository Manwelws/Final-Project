import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


def test_evaluation_report_single_class_guard():
    # Only single class present (all 0s)
    y_true = np.array([0, 0, 0, 0])
    y_probs = np.array([0.1, 0.2, 0.3, 0.4])

    report = EvaluationReport.compute(y_true=y_true, y_probs=y_probs)
    assert report.roc_auc is None
    assert report.pr_auc is None
    assert report.brier_score is None
    assert report.optimization.cost_optimal_threshold > 0.0


def test_optimize_threshold_and_metadata_rounded_to_4_decimals(tmp_path):
    # Probabilities chosen so threshold 0.36 (index 35 of np.arange(0.01, 1.0, 0.01))
    # is the minimum-cost threshold (0.36000000000000004 without rounding)
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.35, 0.37, 0.8])

    opt = optimize_threshold(y_true, y_probs, cost_fn=200.0, cost_fp=20.0)
    assert opt.cost_optimal_threshold == 0.36
    assert opt.f1_optimal_threshold == round(opt.f1_optimal_threshold, 4)
    assert all(m.threshold == round(m.threshold, 4) for m in opt.thresholds_evaluated)

    # Also verify run_evaluation() persists cleanly rounded thresholds to metadata.json
    from sklearn.linear_model import LogisticRegression

    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()

    X_mat, y_vec = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.75, 0.25],
        random_state=42,
    )
    cols = [f"f_{i}" for i in range(4)]
    pd.DataFrame(X_mat, columns=cols).to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_mat, y_vec)
    joblib.dump(clf, models_dir / "model.joblib")

    run_evaluation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        metadata_path=models_dir / "metadata.json",
        report_dir=reports_dir,
        random_state=42,
    )

    meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["optimal_threshold"] == round(meta["optimal_threshold"], 4)
    for key in ("default_0_50", "f1_optimal", "cost_optimal"):
        t_val = meta["threshold_comparison"][key]["threshold"]
        assert t_val == round(t_val, 4)


class _LeakCheckEstimator(BaseEstimator, ClassifierMixin):
    """Test estimator whose predicted probability scale depends on training sample count."""

    def __init__(self, train_only_limit: int = 70):
        self.train_only_limit = train_only_limit

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_sample_count_ = len(X)
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        is_pos = X_arr[:, 0] > 0.5
        if getattr(self, "fitted_sample_count_", 0) > self.train_only_limit:
            # Simulates post-refit (train+val) model: cutoffs around 0.81
            p1 = np.where(is_pos, 0.90, 0.80)
        else:
            # Cloned model fitted ONLY on X_train: cutoffs around 0.21
            p1 = np.where(is_pos, 0.40, 0.20)
        return np.column_stack([1.0 - p1, p1])


class _NonClonableEstimator:
    """Mock estimator without sklearn get_params/fit support."""

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        p1 = np.where(X_arr[:, 0] > 0.5, 0.8, 0.2)
        return np.column_stack([1.0 - p1, p1])


class _UnfittableClonableEstimator(BaseEstimator, ClassifierMixin):
    """Mock estimator that supports clone() but fails during fit()."""

    def __init__(self, pos_prob: float = 0.8, neg_prob: float = 0.2):
        self.pos_prob = pos_prob
        self.neg_prob = neg_prob
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        raise RuntimeError("Refitting is not supported for this mock estimator")

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        p1 = np.where(X_arr[:, 0] > 0.5, self.pos_prob, self.neg_prob)
        return np.column_stack([1.0 - p1, p1])


def test_run_evaluation_calibrates_thresholds_leak_free_on_cloned_train_model(tmp_path, monkeypatch):
    import final_project.evaluation as eval_mod
    from final_project.data import load_and_split_data

    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()

    rng = np.random.RandomState(42)
    y_vec = np.array([0] * 75 + [1] * 25)
    # Feature 0 encodes ground truth cleanly so threshold selection is deterministic
    X_df = pd.DataFrame({
        "signal": y_vec.astype(float),
        "noise": rng.normal(size=100),
    })
    X_df.to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    # Fit serialized model on train+val (80 samples > 70 train-only limit)
    split = load_and_split_data(
        data_dir / "X_features.csv",
        data_dir / "y_target.csv",
        test_size=0.20,
        val_size=0.10,
        random_state=42,
    )
    X_train_val = pd.concat([split.X_train, split.X_val])
    y_train_val = pd.concat([split.y_train, split.y_val])

    clf = _LeakCheckEstimator(train_only_limit=len(split.X_train))
    clf.fit(X_train_val, y_train_val)
    joblib.dump(clf, models_dir / "model.joblib")

    fit_records = []
    orig_clone = eval_mod.clone

    def spy_clone(estimator, safe=True):
        cloned = orig_clone(estimator, safe=safe)
        orig_fit = cloned.fit

        def wrapped_fit(X, y, *args, **kwargs):
            fit_records.append((X.copy(), y.copy()))
            return orig_fit(X, y, *args, **kwargs)

        cloned.fit = wrapped_fit
        return cloned

    monkeypatch.setattr(eval_mod, "clone", spy_clone)

    run_evaluation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        metadata_path=models_dir / "metadata.json",
        report_dir=reports_dir,
        random_state=42,
    )

    # Cloned estimator must have been fitted ONLY on (split.X_train, split.y_train)
    assert len(fit_records) == 1
    fit_X, fit_y = fit_records[0]
    pd.testing.assert_frame_equal(fit_X, split.X_train)
    pd.testing.assert_series_equal(fit_y, split.y_train)

    # Cutoffs must come from cloned train-only model (0.21), NOT post-refit model (0.81)
    meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["optimal_threshold"] == 0.21
    assert meta["threshold_comparison"]["f1_optimal"]["threshold"] == 0.21


@pytest.mark.parametrize("estimator", [_NonClonableEstimator(), _UnfittableClonableEstimator()])
def test_run_evaluation_fallback_on_non_clonable_or_unfittable_estimator(tmp_path, estimator):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()

    y_vec = np.array([0] * 75 + [1] * 25)
    X_df = pd.DataFrame({"signal": y_vec.astype(float), "noise": np.linspace(0, 1, 100)})
    X_df.to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    joblib.dump(estimator, models_dir / "model.joblib")

    summary = run_evaluation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        metadata_path=models_dir / "metadata.json",
        report_dir=reports_dir,
        random_state=42,
    )
    assert isinstance(summary, EvaluationSummary)
    assert 0.0 < summary.optimal_threshold < 1.0


def test_run_evaluation_fallback_when_y_train_is_single_class(tmp_path, monkeypatch):
    import final_project.evaluation as eval_mod
    from final_project.data import DataSplit

    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()

    X_mat, y_vec = make_classification(
        n_samples=60,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X_df = pd.DataFrame(X_mat, columns=[f"f_{i}" for i in range(4)])
    X_df.to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_df, y_vec)
    joblib.dump(clf, models_dir / "model.joblib")

    # Force a DataSplit whose y_train is single-class (all 0s), which would raise in LogisticRegression.fit
    single_class_split = DataSplit(
        X_train=X_df.iloc[:40],
        y_train=pd.Series(np.zeros(40, dtype=int)),
        X_val=X_df.iloc[40:50],
        y_val=pd.Series(y_vec[40:50]),
        X_test=X_df.iloc[50:],
        y_test=pd.Series(y_vec[50:]),
    )
    monkeypatch.setattr(eval_mod, "load_and_split_data", lambda **kwargs: single_class_split)

    summary = run_evaluation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        metadata_path=models_dir / "metadata.json",
        report_dir=reports_dir,
        random_state=42,
    )
    assert isinstance(summary, EvaluationSummary)
    assert 0.0 < summary.optimal_threshold < 1.0


def test_run_evaluation_respects_cost_fn_and_cost_fp(tmp_path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()

    X_mat, y_vec = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        weights=[0.7, 0.3],
        random_state=42,
    )
    X_df = pd.DataFrame(X_mat, columns=[f"f_{i}" for i in range(5)])
    X_df.to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_df, y_vec)
    joblib.dump(clf, models_dir / "model.joblib")

    high_fn_summary = run_evaluation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        metadata_path=models_dir / "metadata_fn.json",
        report_dir=tmp_path / "rep_fn",
        cost_fn=500.0,
        cost_fp=10.0,
        random_state=42,
    )
    high_fp_summary = run_evaluation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        metadata_path=models_dir / "metadata_fp.json",
        report_dir=tmp_path / "rep_fp",
        cost_fn=10.0,
        cost_fp=500.0,
        random_state=42,
    )
    assert high_fn_summary.optimal_threshold < high_fp_summary.optimal_threshold


def test_evaluation_report_context_fields_and_defaults():
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9])

    default_report = EvaluationReport.compute(y_true=y_true, y_probs=y_probs)
    assert default_report.algorithm == ""
    assert default_report.balanced is None
    assert default_report.train_samples is None
    assert default_report.val_samples is None

    ctx_report = EvaluationReport.compute(
        y_true=y_true,
        y_probs=y_probs,
        algorithm="LightGBM",
        balanced=True,
        train_samples=700,
        val_samples=100,
    )
    assert ctx_report.algorithm == "LightGBM"
    assert ctx_report.balanced is True
    assert ctx_report.train_samples == 700
    assert ctx_report.val_samples == 100


def test_run_evaluation_passes_metadata_context_to_report_and_preserves_metadata(tmp_path, monkeypatch):
    import final_project.evaluation as eval_mod

    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()

    X_mat, y_vec = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.7, 0.3],
        random_state=42,
    )
    feature_cols = [f"f_{i}" for i in range(4)]
    X_df = pd.DataFrame(X_mat, columns=feature_cols)
    X_df.to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_df, y_vec)
    joblib.dump(clf, models_dir / "model.joblib")

    initial_meta = {
        "algorithm": "LightGBM (Tuned)",
        "balanced": False,
        "features": feature_cols,
        "cv_auc_mean": 0.91,
        "model_sha256": "abc123",
    }
    (models_dir / "metadata.json").write_text(json.dumps(initial_meta), encoding="utf-8")

    captured_reports = []
    orig_compute = EvaluationReport.compute

    def spy_compute(*args, **kwargs):
        rep = orig_compute(*args, **kwargs)
        captured_reports.append(rep)
        return rep

    monkeypatch.setattr(eval_mod.EvaluationReport, "compute", spy_compute)

    run_evaluation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        metadata_path=models_dir / "metadata.json",
        report_dir=reports_dir,
        test_size=0.20,
        val_size=0.10,
        random_state=42,
    )

    assert len(captured_reports) == 1
    rep = captured_reports[0]
    assert rep.algorithm == "LightGBM (Tuned)"
    assert rep.balanced is False
    assert rep.train_samples == 70
    assert rep.val_samples == 10

    report_text = (reports_dir / "evaluation_report.md").read_text(encoding="utf-8")
    assert "LightGBM (Tuned)" in report_text

    updated_meta = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
    assert updated_meta["algorithm"] == "LightGBM (Tuned)"
    assert updated_meta["balanced"] is False
    assert updated_meta["cv_auc_mean"] == 0.91
    assert updated_meta["model_sha256"] == "abc123"
    assert "optimal_threshold" in updated_meta


def test_run_evaluation_context_defaults_without_metadata_or_val_split(tmp_path, monkeypatch):
    import final_project.evaluation as eval_mod

    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    models_dir.mkdir()

    X_mat, y_vec = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X_df = pd.DataFrame(X_mat, columns=[f"f_{i}" for i in range(4)])
    X_df.to_csv(data_dir / "X_features.csv", index=False)
    pd.DataFrame({"Churn Value": y_vec}).to_csv(data_dir / "y_target.csv", index=False)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_df, y_vec)
    joblib.dump(clf, models_dir / "model.joblib")

    captured_reports = []
    orig_compute = EvaluationReport.compute

    def spy_compute(*args, **kwargs):
        rep = orig_compute(*args, **kwargs)
        captured_reports.append(rep)
        return rep

    monkeypatch.setattr(eval_mod.EvaluationReport, "compute", spy_compute)

    run_evaluation(
        data_dir=data_dir,
        model_path=models_dir / "model.joblib",
        metadata_path=models_dir / "missing_metadata.json",
        report_dir=reports_dir,
        test_size=0.20,
        val_size=0.0,
        random_state=42,
    )

    assert len(captured_reports) == 1
    rep = captured_reports[0]
    assert rep.algorithm == "LogisticRegression"
    assert rep.balanced is None
    assert rep.train_samples == 80
    assert rep.val_samples == 0



