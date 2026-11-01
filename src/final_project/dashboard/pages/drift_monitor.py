"""Page 3: Drift & Retraining Monitor (ADR 0001 Closed-Loop Operations)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from final_project.dashboard.state import (
    load_drift_metrics,
    load_retraining_logs,
    load_feature_psi_matrix,
)
from final_project.dashboard.styles import (
    apply_custom_styles,
    render_hero,
    render_metric_card,
    style_plotly_figure,
    COLORS,
)


def generate_drift_timeline_chart(drift_df: pd.DataFrame) -> go.Figure:
    """Generate dual-axis 8-week timeline of live AUC and Population Stability Index (PSI)."""
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"secondary_y": True}]],
    )

    if drift_df.empty:
        return style_plotly_figure(fig)

    weeks = drift_df["week"].tolist()
    live_auc = drift_df["retrain_model_auc_recent"].tolist()
    ref_auc = drift_df["never_retrain_auc_reference"].tolist()
    psi_mean = drift_df["psi_mean"].tolist()
    retrained = drift_df["retrained"].tolist()

    # 1. Retrain Model Live AUC trace
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=live_auc,
            mode="lines+markers",
            name="Adaptive Retraining AUC (Recent Live)",
            line=dict(color=COLORS["success"], width=3),
            marker=dict(size=8, color=COLORS["success"]),
        ),
        secondary_y=False,
    )

    # 2. Historical Best-Ever Live AUC (Threshold Benchmark)
    if "best_retrain_auc" in drift_df.columns and drift_df["best_retrain_auc"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=drift_df["best_retrain_auc"].tolist(),
                mode="lines",
                name="Historical Best Live AUC Benchmark",
                line=dict(color="#F59E0B", width=2, dash="dot"),
            ),
            secondary_y=False,
        )

    # 3. Never-retrain baseline AUC trace
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=ref_auc,
            mode="lines+markers",
            name="Never-Retrain Baseline AUC (Static Ref)",
            line=dict(color=COLORS["primary"], width=2, dash="dash"),
            marker=dict(size=6, color=COLORS["primary"]),
        ),
        secondary_y=False,
    )

    # 3. Unsupervised Mean PSI on secondary axis
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=psi_mean,
            mode="lines+markers",
            name="Unsupervised Drift (Mean PSI)",
            line=dict(color=COLORS["warning"], width=2, dash="dot"),
            marker=dict(size=6, color=COLORS["warning"]),
        ),
        secondary_y=True,
    )

    # 4. Highlight Retraining Events
    retrain_weeks = [w for w, r in zip(weeks, retrained) if r]
    retrain_aucs = [live_auc[w - 1] for w in retrain_weeks if live_auc[w - 1] is not None]

    fig.add_trace(
        go.Scatter(
            x=retrain_weeks,
            y=retrain_aucs,
            mode="markers",
            name="Retraining Event Fired",
            marker=dict(
                size=14,
                color="#EF4444",
                symbol="triangle-up",
                line=dict(width=2, color="#FFFFFF"),
            ),
        ),
        secondary_y=False,
    )

    # Vertical line indicating drift onset at Week 5
    fig.add_vline(
        x=5,
        line_width=1.5,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Covariate & Concept Drift Onset",
        annotation_position="top left",
        annotation_font=dict(color="#EF4444", size=11),
    )

    fig.update_xaxes(title_text="Simulation Week", dtick=1)
    fig.update_yaxes(title_text="ROC-AUC Score", secondary_y=False, range=[0.75, 1.05])
    fig.update_yaxes(title_text="Mean PSI", secondary_y=True, range=[0.0, 0.35])

    fig.update_layout(
        title="8-Week Streaming Retraining Timeline (ADR 0001 Closed-Loop)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return style_plotly_figure(fig)


def generate_psi_heatmap(psi_matrix: pd.DataFrame) -> go.Figure:
    """Generate 27-feature x 8-week PSI drift heatmap."""
    if psi_matrix.empty:
        return style_plotly_figure(go.Figure())

    # Sort features by max PSI across weeks to bring drifted features to the top
    max_psi = psi_matrix.max(axis=1)
    sorted_features = max_psi.sort_values(ascending=True).index.tolist()
    sorted_df = psi_matrix.loc[sorted_features]

    z_vals = sorted_df.values
    x_labels = sorted_df.columns.tolist()
    y_labels = sorted_df.index.tolist()

    text_vals = np.round(z_vals, 3).astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=x_labels,
            y=y_labels,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#FFFFFF"),
            colorscale=[
                [0.0, "#0F172A"],
                [0.05, "#065F46"],
                [0.15, "#D97706"],
                [1.0, "#DC2626"],
            ],
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title="PSI",
                tickvals=[0.0, 0.10, 0.25, 1.0],
                ticktext=["0.0 (Base)", "0.10 (Stable)", "0.25 (Drift)", ">1.0 (Severe)"],
            ),
        )
    )

    fig.update_layout(
        title="Feature-by-Week Population Stability Index (PSI) Heatmap",
        height=620,
        margin=dict(l=180, r=40, t=50, b=40),
        xaxis=dict(side="bottom"),
    )
    return style_plotly_figure(fig)


def generate_sample_decay_curve(decay_rate: float = 0.85, n_weeks: int = 8) -> go.Figure:
    """Generate exponential sample decay weighting curve (w = decay_rate^age)."""
    ages = np.arange(0, n_weeks + 1)
    weights = decay_rate**ages

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=weights,
            mode="lines+markers",
            name=f"Sample Weight (Rate = {decay_rate})",
            line=dict(color=COLORS["primary"], width=2.5),
            marker=dict(size=8, color=COLORS["secondary"]),
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.12)",
        )
    )

    fig.update_layout(
        title="Exponential Sample Decay Weighting (ADR 0001)",
        xaxis_title="Sample Age in Weeks (Current Week = 0)",
        yaxis_title="Retraining Sample Weight (w)",
        height=340,
        xaxis=dict(dtick=1),
        yaxis=dict(range=[0, 1.1]),
    )
    return style_plotly_figure(fig)


def render_drift_monitor_page() -> None:
    """Render the Drift & Retraining Monitor dashboard page."""
    render_hero(
        title="Streaming Drift & Retraining Operations",
        subtitle="Live 8-week production simulation tracking covariate drift, Population Stability Index (PSI), and adaptive retraining triggers under ADR 0001.",
        badge="ADR 0001 Operations",
    )

    # 1. Load simulation artifacts
    drift_df = load_drift_metrics()
    retrain_logs = load_retraining_logs()
    psi_matrix = load_feature_psi_matrix()

    if drift_df.empty:
        st.info("ℹ️ No drift metrics found in `data/drift/drift_metrics.csv`. Run `churn simulate` or `python notebooks/simulate_retraining.py` to produce weekly streaming metrics.")
        return

    # 2. Executive Stat Cards
    total_weeks = len(drift_df)
    retrain_count = int(drift_df["retrained"].sum()) if "retrained" in drift_df.columns else 0
    max_psi_overall = float(drift_df["psi_max"].max()) if "psi_max" in drift_df.columns else 0.0
    latest_auc = drift_df["retrain_model_auc_recent"].dropna().iloc[-1] if not drift_df["retrain_model_auc_recent"].dropna().empty else 0.84

    # Dynamically compute deltas from data
    retrain_weeks = drift_df[drift_df["retrained"] == True]["week"].tolist() if "retrained" in drift_df.columns else []
    retrain_delta = f"Weeks {retrain_weeks}" if retrain_weeks else "No retrain events"

    drift_weeks = drift_df[drift_df["psi_status"] == "Significant Shift (Drift Detected)"]["week"].tolist() if "psi_status" in drift_df.columns else []
    psi_delta = f"Week {drift_weeks[0]} Onset (Drift)" if drift_weeks else "No Significant Drift"

    if "never_retrain_auc_reference" in drift_df.columns and not drift_df["never_retrain_auc_reference"].dropna().empty:
        never_auc = float(drift_df["never_retrain_auc_reference"].dropna().iloc[-1])
        auc_diff = latest_auc - never_auc
        auc_delta = f"{auc_diff:+.3f} vs Never-Retrain"
    else:
        auc_delta = "Baseline Unavailable"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            label="Streaming Timeline",
            value=f"{total_weeks} Weeks",
            delta="2-Week Label Lag",
            delta_variant="neutral",
            subtext="Weekly batch cycle",
            icon="⏱️",
        )
    with c2:
        render_metric_card(
            label="Retraining Triggers",
            value=f"{retrain_count} Events",
            delta=retrain_delta,
            delta_variant="positive",
            subtext="Dual drift monitoring",
            icon="🔄",
        )
    with c3:
        render_metric_card(
            label="Peak Feature PSI",
            value=f"{max_psi_overall:.2f}",
            delta=psi_delta,
            delta_variant="negative" if drift_weeks else "neutral",
            subtext="Monthly Charges shift",
            icon="⚠️",
        )
    with c4:
        render_metric_card(
            label="Live Retrained AUC",
            value=f"{latest_auc:.4f}",
            delta=auc_delta,
            delta_variant="positive" if "never_auc" in locals() and latest_auc >= never_auc else "neutral",
            subtext="Post-drift recovery",
            icon="🎯",
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 3. 8-Week Streaming Retraining Timeline
    st.markdown("#### 📈 Dual Monitoring Timeline (ADR 0001)")
    st.caption("Visualizing live rolling AUC performance decay alongside unsupervised Population Stability Index (PSI).")
    fig_timeline = generate_drift_timeline_chart(drift_df)
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 4. Tabs: Feature PSI Heatmap & Sample Decay Weighting
    tab1, tab2, tab3 = st.tabs(["🔥 Feature PSI Heatmap", "⚖️ Exponential Sample Decay", "📋 Retraining Event Audit"])

    with tab1:
        st.markdown("##### 🔍 27-Feature Distribution Stability Heatmap across Weeks 1–8")
        st.caption("Green (< 0.10) = Stable; Amber (0.10–0.25) = Moderate Shift; Red (>= 0.25) = Significant Covariate Drift.")
        if not psi_matrix.empty:
            fig_heatmap = generate_psi_heatmap(psi_matrix)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Feature-level weekly data not available for heatmap.")

    with tab2:
        col_decay, col_text = st.columns([3, 2])
        with col_decay:
            decay_rate_slider = st.slider("Exponential Decay Factor (rate)", 0.50, 0.99, 0.85, 0.05)
            fig_decay = generate_sample_decay_curve(decay_rate=decay_rate_slider)
            st.plotly_chart(fig_decay, use_container_width=True)
        with col_text:
            st.markdown("##### 🧠 ADR 0001 Decision Context")
            st.markdown(
                f"""
                - **The Challenge**: After covariate/concept drift at Week 5, older samples reflect stale data distributions.
                - **Why Not a Sliding Window?** Sliding windows drop sample size drastically, leading to high-variance models.
                - **Why Not an Expanding Window?** Expanding windows dilute recent drifted patterns with years of obsolete records.
                - **The Solution (0.85^weeks_old)**: Every historical sample is retained, but exponentially discounted based on its age.
                - **Operational Impact**: At decay rate **{decay_rate_slider}**, 4-week-old samples carry a weight of **{decay_rate_slider**4:.1%}**, allowing newly drifted relationships to guide the decision boundaries.

                """
            )

    with tab3:
        st.markdown("##### 📋 Retraining Trigger Event Logs")
        display_df = drift_df[
            [
                "week",
                "n_labeled",
                "psi_mean",
                "psi_max",
                "psi_status",
                "never_retrain_auc_reference",
                "retrain_model_auc_recent",
                "retrained",
                "trigger_reason",
            ]
        ].copy()
        display_df.columns = [
            "Week",
            "Labeled N",
            "Mean PSI",
            "Max PSI",
            "PSI Status",
            "Baseline Ref AUC",
            "Retrained Live AUC",
            "Retrained?",
            "Decision Trigger Reason",
        ]
        st.dataframe(display_df, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    apply_custom_styles()
    render_drift_monitor_page()
