"""Automated test suite for ChurnOps dashboard state, styling, and KPI computations."""

import numpy as np
import pandas as pd
import pytest

from final_project.domain.cost import CostMatrix
from final_project.dashboard.styles import (
    COLORS,
    CUSTOM_CSS,
    get_plotly_layout_defaults,
    style_plotly_figure,
)
from final_project.dashboard.state import (
    load_clean_dataset,
    load_model_bundle,
    load_metadata,
    calculate_confusion_at_cutoff,
    compute_holdout_financial_impact,
)
import plotly.graph_objects as go


def test_styles_and_theme_tokens() -> None:
    """Verify theme tokens and custom CSS definitions."""
    assert "bg_dark" in COLORS
    assert "primary" in COLORS
    assert "success" in COLORS
    assert "danger" in COLORS

    assert "font-family: 'Outfit'" in CUSTOM_CSS or "Outfit" in CUSTOM_CSS
    assert "glass-card" in CUSTOM_CSS
    assert "metric-value" in CUSTOM_CSS


def test_plotly_style_helper() -> None:
    """Verify Plotly figure dark-mode styling helper."""
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
    styled_fig = style_plotly_figure(fig)
    assert styled_fig.layout.template is not None
    assert styled_fig.layout.paper_bgcolor == "rgba(17, 24, 39, 0)"



def test_load_clean_dataset() -> None:
    """Verify authoritative dataset loader returns valid data structures."""
    X, y, full_df = load_clean_dataset("data/processed")
    assert len(X) == 7043
    assert len(y) == 7043
    assert X.shape[1] == 27
    assert y.nunique() == 2
    assert full_df is not None


def test_load_model_bundle_and_metadata() -> None:
    """Verify production ModelBundle and metadata loading."""
    bundle = load_model_bundle("models")
    meta = load_metadata("models")
    assert bundle is not None
    assert bundle.metadata.algorithm in ["RandomForestClassifier", "LGBMClassifier", "XGBClassifier"]
    assert bundle.metadata.optimal_threshold > 0.0
    assert meta.get("feature_count") == 27


def test_calculate_confusion_at_cutoff() -> None:
    """Verify confusion matrix and cost computation at specific cutoff."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.4, 0.8, 0.1, 0.2])
    cost_matrix = CostMatrix(cost_fn=200.0, cost_fp=20.0)

    # Cutoff = 0.5:
    # predictions: [1, 0, 1, 0, 0]
    # TP: idx 0 (1) -> 1
    # FP: idx 2 (0) -> 1
    # FN: idx 1, 4 (1) -> 2
    # TN: idx 3 (0) -> 1
    res = calculate_confusion_at_cutoff(y_true, y_prob, cutoff=0.5, cost_matrix=cost_matrix)
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["fn"] == 2
    assert res["tn"] == 1
    assert res["total_cost"] == (2 * 200.0) + (1 * 20.0)  # 420.0
    assert res["recall"] == pytest.approx(1 / 3)


def test_compute_holdout_financial_impact() -> None:
    """Verify holdout financial comparison matches expected business logic."""
    bundle = load_model_bundle("models")
    X, y, _ = load_clean_dataset("data/processed")

    from final_project.data import get_stratified_split

    _, X_test, _, y_test = get_stratified_split(X, y)

    impact = compute_holdout_financial_impact(
        bundle=bundle,
        X_test=X_test,
        y_test=y_test,
        cost_fn=200.0,
        cost_fp=20.0,
        optimal_cutoff=0.12,
    )

    assert impact["net_savings"] > 15000.0  # Around $21,860
    assert impact["pct_reduction"] > 50.0  # Around 59.5%
    assert impact["captured_delta"] > 100  # Captures substantially more churners
    assert impact["optimal"]["recall"] > 0.90  # Recall around 93%


def test_candidate_benchmarks_table() -> None:
    """Verify candidate benchmark summary dataframe."""
    from final_project.dashboard.pages.model_diagnostics import get_candidate_benchmarks

    df = get_candidate_benchmarks()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert "Model" in df.columns
    assert "CV ROC-AUC" in df.columns
    assert "LightGBM" in df["Model"].values
    assert "Random Forest" in df["Model"].values
    assert "XGBoost" in df["Model"].values


def test_diagnostic_curve_generators() -> None:
    """Verify Plotly figure generation for ROC, PR, Cost curve, Confusion Matrix, and Calibration."""
    from final_project.dashboard.pages.model_diagnostics import (
        generate_roc_curve,
        generate_pr_curve,
        generate_cost_curve,
        generate_confusion_matrix_heatmap,
        generate_calibration_curve,
    )

    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.85])

    # 1. ROC
    fig_roc = generate_roc_curve(y_true, y_prob)
    assert len(fig_roc.data) == 2
    assert "ROC" in fig_roc.layout.title.text

    # 2. PR
    fig_pr = generate_pr_curve(y_true, y_prob, optimal_cutoff=0.12)
    assert len(fig_pr.data) >= 2
    assert "Precision-Recall" in fig_pr.layout.title.text

    # 3. Cost
    fig_cost = generate_cost_curve(y_true, y_prob, cost_fn=200.0, cost_fp=20.0, optimal_cutoff=0.12)
    assert len(fig_cost.data) == 3
    assert "Cost" in fig_cost.layout.title.text

    # 4. Confusion Matrix Heatmap
    fig_cm = generate_confusion_matrix_heatmap(tp=348, fp=485, tn=550, fn=26, normalize=False)
    assert len(fig_cm.data) == 1
    assert fig_cm.data[0].type == "heatmap"

    fig_cm_norm = generate_confusion_matrix_heatmap(tp=348, fp=485, tn=550, fn=26, normalize=True)
    assert len(fig_cm_norm.data) == 1

    # 5. Calibration
    fig_calib = generate_calibration_curve(y_true, y_prob, n_bins=5)
    assert len(fig_calib.data) == 2
    assert "Calibration" in fig_calib.layout.title.text


def test_drift_monitor_components() -> None:
    """Verify drift timeline, PSI heatmap, and sample decay figure generators."""
    from final_project.dashboard.state import load_drift_metrics, load_feature_psi_matrix
    from final_project.dashboard.pages.drift_monitor import (
        generate_drift_timeline_chart,
        generate_psi_heatmap,
        generate_sample_decay_curve,
    )

    # 1. Timeline chart
    drift_df = load_drift_metrics("data/drift/drift_metrics.csv")
    assert not drift_df.empty
    assert len(drift_df) == 8

    fig_timeline = generate_drift_timeline_chart(drift_df)
    assert len(fig_timeline.data) >= 3
    assert "Streaming Retraining Timeline" in fig_timeline.layout.title.text

    # 2. Feature PSI Matrix & Heatmap
    psi_matrix = load_feature_psi_matrix("data/reference/X_reference.csv", "data/live/features")
    assert not psi_matrix.empty
    assert psi_matrix.shape[1] == 8  # 8 weeks
    assert "Monthly Charges" in psi_matrix.index

    fig_heatmap = generate_psi_heatmap(psi_matrix)
    assert len(fig_heatmap.data) == 1
    assert fig_heatmap.data[0].type == "heatmap"

    # 3. Sample Decay curve
    fig_decay = generate_sample_decay_curve(decay_rate=0.85, n_weeks=8)
    assert len(fig_decay.data) == 1
    assert "Sample Decay" in fig_decay.layout.title.text


def test_retention_triage_components() -> None:
    """Verify customer triage presets, scoring, risk gauge, SHAP bars, and recommendations."""
    from final_project.dashboard.pages.retention_triage import (
        CUSTOMER_PRESETS,
        get_preset_profile,
        score_customer_profile,
        generate_risk_gauge,
        generate_attribution_bar_chart,
        generate_retention_recommendations,
    )
    from final_project.explainability import explain_customer_instance
    from final_project.dashboard.state import load_model_bundle

    # 1. Presets retrieval
    assert "High Risk New Customer" in CUSTOMER_PRESETS
    assert "Loyal Long-Term Customer" in CUSTOMER_PRESETS
    high_risk_preset = get_preset_profile("High Risk New Customer")
    loyal_preset = get_preset_profile("Loyal Long-Term Customer")
    assert high_risk_preset["contract"] == "Month-to-month"
    assert loyal_preset["contract"] == "Two year"

    # 2. Score Customer Profile
    bundle = load_model_bundle("models")
    prob_hi, pred_hi, tier_hi, cutoff_hi, df_hi = score_customer_profile(high_risk_preset, bundle=bundle)
    assert 0.0 <= prob_hi <= 1.0
    assert prob_hi >= cutoff_hi
    assert pred_hi == 1
    assert tier_hi in ("Medium", "High")
    assert df_hi.shape == (1, 27)

    prob_lo, pred_lo, tier_lo, cutoff_lo, df_lo = score_customer_profile(loyal_preset, bundle=bundle)
    assert prob_lo < cutoff_lo
    assert pred_lo == 0
    assert tier_lo == "Low"

    # 3. Speedometer Risk Gauge
    fig_gauge_hi = generate_risk_gauge(prob_hi, threshold=cutoff_hi)
    assert len(fig_gauge_hi.data) == 1
    assert fig_gauge_hi.data[0].type == "indicator"
    assert fig_gauge_hi.data[0].value == round(prob_hi * 100, 1)

    fig_gauge_lo = generate_risk_gauge(prob_lo, threshold=cutoff_lo)
    assert fig_gauge_lo.data[0].gauge.bar.color == "#10B981"

    # 4. Explainability & Signed SHAP Attribution Chart
    explanation = explain_customer_instance(bundle.model, df_hi)
    assert len(explanation.risk_drivers) > 0
    fig_attrib = generate_attribution_bar_chart(explanation.risk_drivers, explanation.protective_factors)
    assert len(fig_attrib.data) == 1
    assert fig_attrib.data[0].type == "bar"

    # 5. Contextual Retention Recommendations
    recs = generate_retention_recommendations(high_risk_preset, explanation.risk_drivers)
    assert len(recs) >= 1
    assert any("Contract" in r["strategy"] or "Commitment" in r["strategy"] for r in recs)
    for r in recs:
        assert "priority" in r
        assert "action" in r
        assert "expected_impact" in r



