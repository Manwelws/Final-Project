"""Page 1: Executive KPI Overview & Financial Cost-Utility Economics."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

from final_project.data import get_stratified_split
from final_project.dashboard.state import (
    load_clean_dataset,
    load_model_bundle,
    compute_holdout_financial_impact,
)
from final_project.dashboard.styles import (
    apply_custom_styles,
    render_hero,
    render_metric_card,
    style_plotly_figure,
    COLORS,
)


def build_human_readable_cohort_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format one-hot encoded dataset into human-readable columns for business exploration."""
    display_df = df.copy()

    # Reconstruct Contract Type
    if "Contract_Month-to-month" in display_df.columns:
        contract_conds = [
            display_df["Contract_Month-to-month"] == 1,
            display_df.get("Contract_One year", 0) == 1,
            display_df.get("Contract_Two year", 0) == 1,
        ]
        display_df["Contract"] = np.select(
            contract_conds, ["Month-to-month", "One year", "Two year"], default="Month-to-month"
        )
    else:
        display_df["Contract"] = "Month-to-month"

    # Reconstruct Internet Service
    if "Internet Service_Fiber optic" in display_df.columns:
        internet_conds = [
            display_df["Internet Service_Fiber optic"] == 1,
            display_df.get("Internet Service_DSL", 0) == 1,
            display_df.get("Internet Service_No", 0) == 1,
        ]
        display_df["Internet Service"] = np.select(
            internet_conds, ["Fiber optic", "DSL", "No"], default="No"
        )
    else:
        display_df["Internet Service"] = "No"

    # Reconstruct Payment Method
    if "Payment Method_Electronic check" in display_df.columns:
        payment_conds = [
            display_df["Payment Method_Electronic check"] == 1,
            display_df.get("Payment Method_Credit card (automatic)", 0) == 1,
            display_df.get("Payment Method_Bank transfer (automatic)", 0) == 1,
            display_df.get("Payment Method_Mailed check", 0) == 1,
        ]
        display_df["Payment Method"] = np.select(
            payment_conds,
            ["Electronic check", "Credit card (auto)", "Bank transfer (auto)", "Mailed check"],
            default="Electronic check",
        )
    else:
        display_df["Payment Method"] = "Electronic check"

    # Status and synthetic Customer ID
    display_df["Status"] = np.where(display_df.get("Churn Value", 0) == 1, "Churned", "Retained")
    display_df["Customer ID"] = [f"CUST-{idx+1001}" for idx in range(len(display_df))]

    return display_df


def generate_churn_drivers_chart(df: pd.DataFrame) -> go.Figure:
    """Generate comparative horizontal bar chart showing churn rates across key segments."""
    drivers = []

    if "Contract_Month-to-month" in df.columns:
        m2m = df[df["Contract_Month-to-month"] == 1]["Churn Value"].mean() * 100
        two_yr = df[df["Contract_Two year"] == 1]["Churn Value"].mean() * 100
        drivers.append(("Month-to-Month Contract", m2m, "#EF4444"))
        drivers.append(("Two-Year Contract", two_yr, "#10B981"))

    if "Payment Method_Electronic check" in df.columns:
        echeck = df[df["Payment Method_Electronic check"] == 1]["Churn Value"].mean() * 100
        auto_cc = df[df["Payment Method_Credit card (automatic)"] == 1]["Churn Value"].mean() * 100
        drivers.append(("Payment: Electronic Check", echeck, "#EF4444"))
        drivers.append(("Payment: Credit Card (Auto)", auto_cc, "#10B981"))

    if "Internet Service_Fiber optic" in df.columns:
        fiber = df[df["Internet Service_Fiber optic"] == 1]["Churn Value"].mean() * 100
        dsl = df[df["Internet Service_DSL"] == 1]["Churn Value"].mean() * 100
        drivers.append(("Service: Fiber Optic", fiber, "#EF4444"))
        drivers.append(("Service: DSL", dsl, "#10B981"))

    if "Tech Support" in df.columns:
        no_tech = df[df["Tech Support"] == 0]["Churn Value"].mean() * 100
        has_tech = df[df["Tech Support"] == 1]["Churn Value"].mean() * 100
        drivers.append(("No Tech Support", no_tech, "#F59E0B"))
        drivers.append(("With Tech Support", has_tech, "#10B981"))

    if not drivers:
        return style_plotly_figure(go.Figure())

    labels = [d[0] for d in drivers]
    rates = [d[1] for d in drivers]
    colors = [d[2] for d in drivers]
    text_labels = [f"{r:.1f}%" for r in rates]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=rates,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255, 255, 255, 0.15)", width=1)),
            text=text_labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Observed Churn Rate: %{x:.1f}%<extra></extra>",
        )
    )

    style_plotly_figure(fig)
    fig.update_layout(
        title="Empirical Churn Rate by Customer Segment",
        xaxis_title="Churn Rate (%)",
        yaxis_title="Segment Driver",
        height=380,
        margin=dict(l=170, r=40, t=50, b=40),
        xaxis=dict(range=[0, 55]),
    )
    return fig


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
        logger.error("Failed to load dataset or model bundle: %s", exc, exc_info=True)
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

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)

    # 7. Key Portfolio Churn Drivers & Segment Vulnerability
    st.markdown("### 🧠 Key Portfolio Churn Drivers & Segment Vulnerability")
    st.caption("Empirical churn vulnerability across contract structures, payment options, and telecom services.")

    driver_col1, driver_col2 = st.columns([1.3, 1])

    with driver_col1:
        if full_df is not None:
            fig_drivers = generate_churn_drivers_chart(full_df)
            st.plotly_chart(fig_drivers, use_container_width=True)
        else:
            st.info("Full customer dataset not loaded.")

    with driver_col2:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF; margin-bottom: 0.6rem;">
                    💡 Executive Takeaways
                </div>
                <div style="font-size: 0.85rem; color: #D1D5DB; line-height: 1.6;">
                    <p><b>1. Contract Commitment Gap:</b> Month-to-month subscribers experience an alarming <b>42.7%</b> churn rate compared to only <b>2.8%</b> for two-year contracts.</p>
                    <p><b>2. Payment Friction:</b> Electronic check payers churn at <b>45.3%</b>—nearly 3× higher than automated credit card billing (15.2%).</p>
                    <p><b>3. Tech Support Shield:</b> Accounts without technical support churn at <b>31.2%</b> vs <b>15.2%</b> with support, proving support is a major retention anchor.</p>
                    <p style="color: #60A5FA; margin-bottom: 0;"><b>Action:</b> Shift month-to-month users to annual terms with automatic billing incentives.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)

    # 8. Annualized ROI & Retention Campaign Capacity Simulator
    st.markdown("### 💼 Annualized Retention ROI & Campaign Capacity Simulator")
    st.caption("Simulate portfolio-scale retention campaign economics and projected Net ARR protected.")

    with st.container():
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            target_cohort_size = st.number_input(
                "Targeted At-Risk Customers",
                min_value=100,
                max_value=total_customers,
                value=1200,
                step=100,
            )
        with s_col2:
            save_rate = st.slider(
                "Intervention Save Rate (%)",
                min_value=5,
                max_value=75,
                value=35,
                step=5,
                help="Percentage of targeted churners who accept the offer and remain retained.",
            )
        with s_col3:
            offer_cost = st.number_input(
                "Incentive Offer / Contact ($)",
                min_value=5.0,
                max_value=100.0,
                value=20.0,
                step=5.0,
            )
        with s_col4:
            arr_per_customer = st.number_input(
                "Annual Customer Value ($ ARR)",
                min_value=300.0,
                max_value=5000.0,
                value=1080.0,
                step=60.0,
                help="Annual recurring revenue per retained account ($90/mo average).",
            )

        # Simulation calculations
        saved_churners = int(round(target_cohort_size * (save_rate / 100.0)))
        total_campaign_cost = target_cohort_size * offer_cost
        gross_arr_saved = saved_churners * arr_per_customer
        net_arr_saved = gross_arr_saved - total_campaign_cost
        roi_multiple = (gross_arr_saved / total_campaign_cost) if total_campaign_cost > 0 else 0.0

        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            render_metric_card(
                label="Customers Rescued",
                value=f"{saved_churners:,}",
                delta=f"{save_rate}% Save Rate",
                delta_variant="positive",
                subtext=f"from {target_cohort_size:,} targeted",
                icon="🛡️",
            )
        with r_col2:
            render_metric_card(
                label="Campaign Investment",
                value=f"${total_campaign_cost:,.0f}",
                delta=f"${offer_cost:.0f} / account",
                delta_variant="neutral",
                subtext="Total outreach budget",
                icon="💳",
            )
        with r_col3:
            render_metric_card(
                label="Gross ARR Saved",
                value=f"${gross_arr_saved:,.0f}",
                delta="Annualized Protection",
                delta_variant="positive",
                subtext=f"${arr_per_customer:,.0f} value/account",
                icon="📈",
            )
        with r_col4:
            render_metric_card(
                label="Net Business ROI",
                value=f"${net_arr_saved:,.0f}",
                delta=f"{roi_multiple:.1f}x ROI Return",
                delta_variant="positive",
                subtext="Net annual profit gain",
                icon="🚀",
            )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)

    # 9. Customer Portfolio Cohort Explorer & CSV Export
    st.markdown("### 🔍 Customer Portfolio Cohort Explorer & Data Export")
    st.caption("Filter and export specific customer segments across demographic, service, and payment profiles.")

    if full_df is not None:
        readable_df = build_human_readable_cohort_df(full_df)

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            contracts = ["All"] + sorted(list(readable_df["Contract"].unique()))
            sel_contract = st.selectbox("Contract Type", contracts, index=0)
        with f_col2:
            internets = ["All"] + sorted(list(readable_df["Internet Service"].unique()))
            sel_internet = st.selectbox("Internet Service", internets, index=0)
        with f_col3:
            payments = ["All"] + sorted(list(readable_df["Payment Method"].unique()))
            sel_payment = st.selectbox("Payment Method", payments, index=0)
        with f_col4:
            statuses = ["All", "Churned", "Retained"]
            sel_status = st.selectbox("Customer Status", statuses, index=0)

        # Apply cohort filtering
        filtered_cohort = readable_df.copy()
        if sel_contract != "All":
            filtered_cohort = filtered_cohort[filtered_cohort["Contract"] == sel_contract]
        if sel_internet != "All":
            filtered_cohort = filtered_cohort[filtered_cohort["Internet Service"] == sel_internet]
        if sel_payment != "All":
            filtered_cohort = filtered_cohort[filtered_cohort["Payment Method"] == sel_payment]
        if sel_status != "All":
            filtered_cohort = filtered_cohort[filtered_cohort["Status"] == sel_status]

        cohort_size = len(filtered_cohort)
        cohort_churn_rate = (
            (filtered_cohort["Status"] == "Churned").mean() * 100 if cohort_size > 0 else 0.0
        )
        avg_monthly = filtered_cohort["Monthly Charges"].mean() if cohort_size > 0 else 0.0
        total_mrr = filtered_cohort["Monthly Charges"].sum() if cohort_size > 0 else 0.0

        cf_col1, cf_col2, cf_col3, cf_col4 = st.columns(4)
        with cf_col1:
            st.metric("Cohort Size", f"{cohort_size:,} accounts")
        with cf_col2:
            st.metric("Cohort Churn Rate", f"{cohort_churn_rate:.1f}%")
        with cf_col3:
            st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}/mo")
        with cf_col4:
            st.metric("Total Cohort MRR", f"${total_mrr:,.0f}")

        # Display table columns
        preview_cols = [
            "Customer ID",
            "Contract",
            "Internet Service",
            "Payment Method",
            "Tenure Months",
            "Monthly Charges",
            "Total Charges",
            "CLTV",
            "Status",
        ]
        available_cols = [c for c in preview_cols if c in filtered_cohort.columns]

        st.dataframe(
            filtered_cohort[available_cols].head(150),
            hide_index=True,
            use_container_width=True,
            height=260,
        )

        csv_bytes = filtered_cohort[available_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Cohort (.csv)",
            data=csv_bytes,
            file_name=f"churnops_cohort_{sel_contract.lower().replace(' ', '_')}_{sel_status.lower()}.csv",
            mime="text/csv",
        )
    else:
        st.info("Full customer dataset not found for cohort exploration.")


if __name__ == "__main__":
    apply_custom_styles()
    render_kpi_overview_page()
