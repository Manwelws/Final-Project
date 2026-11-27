"""Tests for evaluation report generation and backward compatibility."""

from types import SimpleNamespace
import numpy as np
import pytest

from final_project.domain.cost import CostMatrix
from final_project.evaluation import optimize_threshold
from final_project.reporting import EvaluationContext, generate_evaluation_report


def _make_opt_result():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    return y_true, y_probs, optimize_threshold(y_true, y_probs, cost_fn=200.0, cost_fp=20.0)


@pytest.mark.parametrize(
    ("balanced", "expected_strategy"),
    [
        (True, "balanced"),
        (False, "unweighted"),
        (None, "Not specified"),
    ],
)
def test_report_displays_algorithm_and_imbalance_strategy_from_report_object(
    tmp_path, balanced, expected_strategy
):
    y_true, y_probs, opt = _make_opt_result()
    report_obj = SimpleNamespace(
        y_true=y_true,
        y_probs=y_probs,
        roc_auc=0.95,
        pr_auc=0.92,
        brier_score=0.08,
        optimization=opt,
        cost_matrix=CostMatrix(cost_fn=200.0, cost_fp=20.0),
        algorithm="LightGBM",
        balanced=balanced,
    )
    out = tmp_path / "evaluation_report.md"
    generate_evaluation_report(report=report_obj, report_file=out)
    text = out.read_text(encoding="utf-8")

    assert "LightGBM" in text
    assert "Imbalance Strategy" in text
    assert expected_strategy in text
    if balanced is None:
        assert "balanced" not in text.lower().split("imbalance strategy")[1].splitlines()[0]
        assert "unweighted" not in text.lower().split("imbalance strategy")[1].splitlines()[0]


def test_report_handles_omitted_algorithm_and_balanced_attributes(tmp_path):
    y_true, y_probs, opt = _make_opt_result()
    minimal_report = SimpleNamespace(
        y_true=y_true,
        y_probs=y_probs,
        roc_auc=0.95,
        pr_auc=0.92,
        brier_score=0.08,
        optimization=opt,
    )
    out = tmp_path / "evaluation_report.md"
    generate_evaluation_report(minimal_report, out)
    text = out.read_text(encoding="utf-8")

    assert "Not specified" in text
    assert "Imbalance Strategy" in text


def test_report_documents_split_calibration_and_refit_methodology(tmp_path):
    y_true, y_probs, opt = _make_opt_result()
    report_obj = SimpleNamespace(
        y_true=y_true,
        y_probs=y_probs,
        roc_auc=0.95,
        pr_auc=0.92,
        brier_score=0.08,
        optimization=opt,
        algorithm="RandomForestClassifier",
        balanced=True,
        train_samples=56,
        val_samples=8,
    )
    out = tmp_path / "evaluation_report.md"
    generate_evaluation_report(report=report_obj, report_file=out)
    text = out.read_text(encoding="utf-8")

    assert "70/10/20" in text
    assert "stratified split" in text.lower()
    assert "validation" in text.lower()
    assert "threshold" in text.lower()
    assert "train+val" in text.lower() or "train + val" in text.lower()
    assert "refit" in text.lower()
    assert "56" in text
    assert "8" in text


def test_report_references_evaluation_figures_and_notes_missing_shap(tmp_path):
    y_true, y_probs, opt = _make_opt_result()
    report_obj = SimpleNamespace(
        y_true=y_true,
        y_probs=y_probs,
        roc_auc=0.95,
        pr_auc=0.92,
        brier_score=0.08,
        optimization=opt,
    )
    out = tmp_path / "evaluation_report.md"
    generate_evaluation_report(report=report_obj, report_file=out)
    text = out.read_text(encoding="utf-8")

    expected_eval_figures = [
        "figures/roc_curve.png",
        "figures/pr_curve.png",
        "figures/calibration_curve.png",
        "figures/cost_curve.png",
        "figures/confusion_matrix.png",
    ]
    for fig_ref in expected_eval_figures:
        assert fig_ref in text

    shap_figures = [
        "figures/shap_summary_grouped.png",
        "figures/shap_summary_detailed.png",
        "figures/shap_dependence_monthly_charges.png",
        "figures/shap_dependence_tenure.png",
        "figures/shap_waterfall_churner.png",
        "figures/shap_waterfall_retained.png",
    ]
    for shap_ref in shap_figures:
        assert shap_ref not in text

    assert "churn explain" in text
    assert "figures/" in text


def test_report_references_existing_shap_figures_only(tmp_path):
    y_true, y_probs, opt = _make_opt_result()
    report_obj = SimpleNamespace(
        y_true=y_true,
        y_probs=y_probs,
        roc_auc=0.95,
        pr_auc=0.92,
        brier_score=0.08,
        optimization=opt,
    )
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    present_shap = [
        "shap_summary_grouped.png",
        "shap_summary_detailed.png",
        "shap_waterfall_churner.png",
    ]
    for name in present_shap:
        (figures_dir / name).write_bytes(b"png")

    out = tmp_path / "evaluation_report.md"
    generate_evaluation_report(report=report_obj, report_file=out)
    text = out.read_text(encoding="utf-8")

    for name in present_shap:
        assert f"figures/{name}" in text

    absent_shap = [
        "shap_dependence_monthly_charges.png",
        "shap_dependence_tenure.png",
        "shap_waterfall_retained.png",
    ]
    for name in absent_shap:
        assert f"figures/{name}" not in text
    assert "churn explain" not in text


def test_report_accepts_evaluation_context_object(tmp_path):
    from final_project.evaluation import EvaluationContext as EvalCtx, EvaluationReport

    assert EvalCtx is EvaluationContext
    y_true, y_probs, _ = _make_opt_result()
    ctx = EvaluationContext(
        algorithm="CatBoost",
        balanced=True,
        train_samples=70,
        val_samples=10,
    )
    report = EvaluationReport.compute(y_true=y_true, y_probs=y_probs, context=ctx)
    out = tmp_path / "context_report.md"
    report.write_markdown(out)
    text = out.read_text(encoding="utf-8")

    assert "CatBoost" in text
    assert "`balanced`" in text
    assert "70 train" in text
    assert "10 validation" in text


def test_report_preserves_legacy_summary_dict_signature(tmp_path):
    _, _, opt = _make_opt_result()
    legacy_summary = {
        "test_samples": 120,
        "roc_auc": 0.84,
        "pr_auc": 0.65,
        "brier_score": 0.14,
    }
    out_legacy = tmp_path / "legacy.md"
    generate_evaluation_report(legacy_summary, opt, out_legacy)
    legacy_text = out_legacy.read_text(encoding="utf-8")

    assert "120 customers" in legacy_text
    assert "`0.8400`" in legacy_text
    assert "Not specified" in legacy_text

    enriched_summary = {
        **legacy_summary,
        "algorithm": "XGBoost",
        "balanced": False,
        "train_samples": 420,
        "val_samples": 60,
    }
    out_enriched = tmp_path / "enriched.md"
    generate_evaluation_report(enriched_summary, opt, out_enriched)
    enriched_text = out_enriched.read_text(encoding="utf-8")

    assert "XGBoost" in enriched_text
    assert "unweighted" in enriched_text
    assert "420" in enriched_text
    assert "60" in enriched_text


def test_report_computes_dynamic_split_percentages_when_val_samples_is_zero(tmp_path):
    y_true = np.array([0, 1] * 10)
    y_probs = np.linspace(0.1, 0.9, 20)
    opt = optimize_threshold(y_true, y_probs, cost_fn=200.0, cost_fp=20.0)
    report_obj = SimpleNamespace(
        y_true=y_true,
        y_probs=y_probs,
        roc_auc=0.95,
        pr_auc=0.92,
        brier_score=0.08,
        optimization=opt,
        algorithm="LightGBM",
        balanced=True,
        train_samples=80,
        val_samples=0,
    )
    out = tmp_path / "no_val_report.md"
    generate_evaluation_report(report=report_obj, report_file=out)
    text = out.read_text(encoding="utf-8")

    exec_summary = text.split("## 2.")[0]
    assert "80 train (80%)" in exec_summary
    assert "20 customers (20% stratified split)" in exec_summary
    assert "(70%)" not in exec_summary
    assert "(10%)" not in exec_summary



