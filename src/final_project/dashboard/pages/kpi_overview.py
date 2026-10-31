"""Page 1: Executive KPI Overview & Financial Cost-Utility Economics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from final_project.data import get_stratified_split
from final_project.dashboard.state import (
    load_clean_dataset,
    load_model_bundle,
    compute_holdout_financial_impact,
)
from final_project.dashboard.styles import (
    render_hero,
    render_metric_card,
    style_plotly_figure,
    COLORS,
)


def render_kpi_overview_page() -> None:
    """Render the executive KPI and financial impact dashboard page."""
    render_hero(
        title="Executive KPI & Financial Economics",
        subtitle="Customer churn portfolio metrics, cost-utility threshold optimization, and unlocked business ROI.",
        badge="Executive Overview",
    )

    # 1. Load data & production model
    try:
        X, y, full_df = load_clean_dataset()
        bundle = load_model_bundle()
    except Exception as exc:
        st.error(f"Failed to load dataset or model bundle: {exc}")
        return

    # Split for holdout evaluation
    _, X_test, _, y_test = get_stratified_split(X, y)

    # 2. Financial Economics Configuration Controls
    with st.expander("⚙️ Financial Cost Matrix Parameters", expanded=False):
        c_col1, c_col2, c_col3 = st.columns([1, 1, 1])
        with c_col1:
            cost_fn = st.slider(
                "Cost of Missed Churner (C_FN)",
                min_value=50.0,
                max_value=500.0,
                value=200.0,
                step=10.0,
                help="Financial penalty when an at-risk customer churns undetected.",
            )
        with c_col2:
            cost_fp = st.slider(
                "Cost of Retention Incentive (C_FP)",
                min_value=5.0,
                max_value=100.0,
                value=20.0,
                step=5.0,
                help="Cost of discount, promotion, or outreach given to a retained customer.",
            )
        with c_col3:
            custom_cutoff = st.slider(
                "Decision Cutoff (Threshold)",
                min_value=0.01,
                max_value=0.99,
                value=float(bundle.metadata.optimal_threshold),
                step=0.01,
                help="Probability cutoff where customer is classified as Churner.",
            )

    # 3. Calculate metrics
    impact = compute_holdout_financial_impact(
        bundle=bundle,
        X_test=X_test,
        y_test=y_test,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        optimal_cutoff=custom_cutoff,
    )

    total_customers = len(y)
    churn_count = int(np.sum(y == 1))
    retained_count = int(np.sum(y == 0))
    churn_rate = (churn_count / total_customers) * 100

    # 4. Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            label="Total Customer Base",
            value=f"{total_customers:,}",
            delta="100% Monitored",
            delta_variant="neutral",
            subtext="Cleaned customer cohort",
            icon="👥",
        )

    with col2:
        render_metric_card(
            label="Base Churn Rate",
            value=f"{churn_rate:.1f}%",
            delta=f"{churn_count:,} Churned",
            delta_variant="negative",
            subtext=f"{retained_count:,} Retained",
            icon="📉",
        )

    with col3:
        render_metric_card(
            label="Net Business Savings",
            value=f"${impact['net_savings']:,.0f}",
            delta=f"-{impact['pct_reduction']:.1f}% Loss Reduction",
            delta_variant="positive",
            subtext=f"vs Default 0.50 Cutoff (${impact['default']['total_cost']:,.0f})",
            icon="💰",
        )

    with col4:
        opt_recall = impact["optimal"]["recall"] * 100
        render_metric_card(
            label="Holdout Churn Recall",
            value=f"{opt_recall:.1f}%",
            delta=f"+{impact['captured_delta']} Churners Rescued",
            delta_variant="positive",
            subtext=f"{impact['optimal']['tp']} of {impact['total_holdout_churners']} Captured",
            icon="🎯",
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 5. Charts: Business Loss Comparison & Customer Portfolio Breakdown
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown("#### 💵 Financial Loss Reduction by Strategy")
        st.caption(
            f"Holdout Test Set ($N=1,409$). Cost matrix: Missed Churn = ${cost_fn:,.0f}, Retention Offer = ${cost_fp:,.0f}."
        )

        def_cost = impact["default"]["total_cost"]
        opt_cost = impact["optimal"]["total_cost"]
        savings = impact["net_savings"]

        def_fn_cost = impact["default"]["fn"] * cost_fn
        def_fp_cost = impact["default"]["fp"] * cost_fp
        opt_fn_cost = impact["optimal"]["fn"] * cost_fn
        opt_fp_cost = impact["optimal"]["fp"] * cost_fp

        fig_loss = go.Figure()

        # Stacked bar for default
        fig_loss.add_trace(
            go.Bar(
                name="Missed Churn Penalty (FN)",
                x=["Default Strategy (0.50)", f"Cost-Optimized ({custom_cutoff:.2f})"],
                y=[def_fn_cost, opt_fn_cost],
                marker_color="#EF4444",
                text=[f"${def_fn_cost:,.0f}", f"${opt_fn_cost:,.0f}"],
                textposition="auto",
            )
        )
        fig_loss.add_trace(
            go.Bar(
                name="Retention Incentive (FP)",
                x=["Default Strategy (0.50)", f"Cost-Optimized ({custom_cutoff:.2f})"],
                y=[def_fp_cost, opt_fp_cost],
                marker_color="#F59E0B",
                text=[f"${def_fp_cost:,.0f}", f"${opt_fp_cost:,.0f}"],
                textposition="auto",
            )
        )

        fig_loss.update_layout(
            barmode="stack",
            height=340,
            yaxis_title="Expected Loss ($ USD)",
            xaxis_title="Decision Strategy",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        style_plotly_figure(fig_loss)
        st.plotly_chart(fig_loss, use_container_width=True)

    with chart_col2:
        st.markdown("#### 📊 Customer Portfolio Breakdown")
        st.caption("Distribution of active vs churned customer relationships.")

        labels = ["Retained Customers", "Churned Customers"]
        values = [retained_count, churn_count]
        colors = ["#10B981", "#EF4444"]

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.62,
                    marker=dict(colors=colors, line=dict(color="#111827", width=2)),
                    textinfo="label+percent",
                    insidetextorientation="radial",
                )
            ]
        )

        fig_pie.update_layout(
            height=340,
            showlegend=False,
            annotations=[
                dict(
                    text=f"<b>{total_customers:,}</b><br><span style='font-size:12px;color:#9CA3AF;'>Accounts</span>",
                    x=0.5,
                    y=0.5,
                    font_size=18,
                    showarrow=False,
                )
            ],
        )
        style_plotly_figure(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 6. Detailed Comparison Table
    st.markdown("#### 📋 Strategy Benchmark Details")
    comp_df = pd.DataFrame(
        [
            {
                "Strategy": "Standard Uncalibrated (Cutoff 0.50)",
                "Cutoff": "0.50",
                "Captured Churners (TP)": f"{impact['default']['tp']:,}",
                "Missed Churners (FN)": f"{impact['default']['fn']:,}",
                "Unnecessary Offers (FP)": f"{impact['default']['fp']:,}",
                "Recall": f"{impact['default']['recall']*100:.1f}%",
                "Precision": f"{impact['default']['precision']*100:.1f}%",
                "Total Business Cost": f"${def_cost:,.0f}",
            },
            {
                "Strategy": f"Cost-Utility Optimized (Cutoff {custom_cutoff:.2f})",
                "Cutoff": f"{custom_cutoff:.2f}",
                "Captured Churners (TP)": f"{impact['optimal']['tp']:,}",
                "Missed Churners (FN)": f"{impact['optimal']['fn']:,}",
                "Unnecessary Offers (FP)": f"{impact['optimal']['fp']:,}",
                "Recall": f"{impact['optimal']['recall']*100:.1f}%",
                "Precision": f"{impact['optimal']['precision']*100:.1f}%",
                "Total Business Cost": f"${opt_cost:,.0f}",
            },
        ]
    )
    st.dataframe(comp_df, hide_index=True, use_container_width=True)
