"""Page 4: Real-Time Customer Retention Scoring, Triage & SHAP Waterfall."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from final_project.serving.schemas import CustomerProfileRaw
from final_project.serving.adapter import encode_raw_profile
from final_project.explainability import explain_customer_instance, CustomerExplanation
from final_project.registry import ModelBundle
from final_project.dashboard.state import load_model_bundle, load_metadata
from final_project.dashboard.styles import (
    render_hero,
    style_plotly_figure,
    COLORS,
)

# Representative Customer Archetype Presets
CUSTOMER_PRESETS: dict[str, dict[str, Any]] = {
    "High Risk New Customer": {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "tenure_months": 2.0,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 89.50,
        "total_charges": 179.00,
        "cltv": 3200.0,
    },
    "Loyal Long-Term Customer": {
        "gender": "Male",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "Yes",
        "tenure_months": 64.0,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless_billing": "No",
        "payment_method": "Credit card (automatic)",
        "monthly_charges": 54.00,
        "total_charges": 3456.00,
        "cltv": 4800.0,
    },
    "Drifted High-Charge Customer": {
        "gender": "Male",
        "senior_citizen": "Yes",
        "partner": "No",
        "dependents": "No",
        "tenure_months": 14.0,
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 118.50,
        "total_charges": 1659.00,
        "cltv": 4100.0,
    },
}


def get_preset_profile(preset_name: str) -> dict[str, Any]:
    """Retrieve a copy of the customer archetype profile dictionary."""
    if preset_name in CUSTOMER_PRESETS:
        return dict(CUSTOMER_PRESETS[preset_name])
    return dict(CUSTOMER_PRESETS["High Risk New Customer"])


def score_customer_profile(
    profile_dict: dict[str, Any],
    bundle: ModelBundle | None = None,
    use_api: bool = False,
    api_url: str = "http://localhost:8000",
) -> tuple[float, int, str, float, pd.DataFrame]:
    """Score raw customer profile and return (prob, binary_pred, risk_tier, cutoff, encoded_df)."""
    df = encode_raw_profile(profile_dict)

    if use_api:
        try:
            import httpx

            resp = httpx.post(f"{api_url}/predict", json=profile_dict, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                return (
                    float(data["churn_probability"]),
                    int(data["churn_prediction"]),
                    str(data["risk_tier"]),
                    float(data["threshold_applied"]),
                    df,
                )
        except Exception:
            pass  # Fallback to local ModelBundle

    if bundle is None:
        bundle = load_model_bundle()

    risk_pred = bundle.predict_risk(df)
    churn_prob = float(risk_pred.probabilities[0])
    churn_pred = int(risk_pred.predictions[0])
    risk_tier = str(risk_pred.risk_tiers[0])
    cutoff = float(risk_pred.threshold_applied)

    return churn_prob, churn_pred, risk_tier, cutoff, df


def generate_risk_gauge(probability: float, threshold: float = 0.12) -> go.Figure:
    """Generate an interactive Plotly Speedometer Gauge with calibrated risk bands."""
    pct_val = round(probability * 100, 1)
    threshold_pct = round(threshold * 100, 1)

    if probability < threshold:
        gauge_bar_color = "#10B981"  # Emerald
    elif probability < 0.30:
        gauge_bar_color = "#F59E0B"  # Amber
    else:
        gauge_bar_color = "#EF4444"  # Crimson

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct_val,
            number={"suffix": "%", "font": {"size": 42, "family": "Outfit, sans-serif"}},
            title={"text": "CHURN RISK PROBABILITY", "font": {"size": 15, "color": "#9CA3AF"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4B5563"},
                "bar": {"color": gauge_bar_color, "thickness": 0.28},
                "bgcolor": "rgba(31, 41, 55, 0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, threshold_pct], "color": "rgba(16, 185, 129, 0.22)"},
                    {"range": [threshold_pct, 30], "color": "rgba(245, 158, 11, 0.22)"},
                    {"range": [30, 100], "color": "rgba(239, 68, 68, 0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#60A5FA", "width": 4},
                    "thickness": 0.85,
                    "value": threshold_pct,
                },
            },
        )
    )

    style_plotly_figure(fig)
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=50, b=20),
        annotations=[
            dict(
                text=f"Decision Cutoff: {threshold_pct:.1f}%",
                x=0.5,
                y=0.15,
                showarrow=False,
                font=dict(size=12, color="#60A5FA"),
            )
        ],
    )
    return fig


def generate_attribution_bar_chart(
    risk_drivers: list[tuple[str, float]],
    protective_factors: list[tuple[str, float]],
    top_n: int = 6,
) -> go.Figure:
    """Generate signed SHAP attribution horizontal bar chart comparing risk vs protective factors."""
    # Top N risk drivers (positive push toward churn)
    selected_drivers = risk_drivers[:top_n]
    # Top N protective factors (negative push away from churn)
    selected_protective = protective_factors[:top_n]

    # Combine factors for unified display
    all_factors = selected_protective[::-1] + selected_drivers

    if not all_factors:
        # Fallback if empty
        all_factors = [("Baseline Model Prior", 0.0)]

    names = [f[0] for f in all_factors]
    values = [f[1] for f in all_factors]
    bar_colors = ["#EF4444" if v >= 0 else "#10B981" for v in values]
    text_labels = [f"+{v:.3f}" if v >= 0 else f"{v:.3f}" for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(255, 255, 255, 0.15)", width=1)),
            text=text_labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>",
        )
    )

    style_plotly_figure(fig)
    fig.update_layout(
        title="Signed Feature Attributions (SHAP TreeExplainer)",
        xaxis_title="Net SHAP Value (Log-Odds Impact on Churn)",
        yaxis_title="Parent Feature Domain",
        height=380,
        margin=dict(l=140, r=40, t=60, b=40),
        xaxis=dict(
            zeroline=True,
            zerolinecolor="#6B7280",
            zerolinewidth=2,
            gridcolor="#1F2937",
        ),
    )
    return fig


def generate_retention_recommendations(
    profile: dict[str, Any],
    risk_drivers: list[tuple[str, float]],
) -> list[dict[str, str]]:
    """Produce actionable retention intervention strategies tailored to the customer's attributes."""
    recommendations: list[dict[str, str]] = []
    driver_names = {item[0] for item in risk_drivers[:6]}

    # 1. Contract Commitment Strategy
    contract_val = str(profile.get("contract", "")).lower()
    if "month-to-month" in contract_val or "Contract" in driver_names:
        recommendations.append(
            {
                "priority": "CRITICAL",
                "badge": "badge-crimson",
                "strategy": "Annual Commitment Transition Offer",
                "action": "Offer a 12-month contract commitment with an exclusive 15% discount for the first 6 months ($12.50/mo bill credit).",
                "expected_impact": "Reduces contract volatility and locks in retention for 12 months, mitigating baseline hazard by ~68%.",
            }
        )

    # 2. Payment Modernization Strategy
    payment_val = str(profile.get("payment_method", "")).lower()
    if "electronic check" in payment_val or "Payment Method" in driver_names:
        recommendations.append(
            {
                "priority": "HIGH",
                "badge": "badge-amber",
                "strategy": "Automated Billing Incentive",
                "action": "Enroll customer in automated ACH or credit card billing with a one-time $25 statement credit.",
                "expected_impact": "Eliminates monthly friction and manual renewal hesitation, reducing payment-induced churn.",
            }
        )

    # 3. Technical Support & Value Shield Strategy
    tech_support = str(profile.get("tech_support", "")).lower()
    online_sec = str(profile.get("online_security", "")).lower()
    if (
        tech_support in ("no", "0", "false")
        or online_sec in ("no", "0", "false")
        or "Tech Support" in driver_names
        or "Online Security" in driver_names
    ):
        recommendations.append(
            {
                "priority": "HIGH",
                "badge": "badge-blue",
                "strategy": "Complimentary Tech & Security Shield",
                "action": "Provision a 6-month free tier of 24/7 Premium Tech Support and Online Security.",
                "expected_impact": "Significantly strengthens product stickiness and prevents early service dissatisfaction.",
            }
        )

    # 4. Plan Right-Sizing Strategy (High Monthly Charges / Fiber Optic)
    monthly_chg = float(profile.get("monthly_charges", 70.0))
    if monthly_chg >= 85.0 or "Monthly Charges" in driver_names:
        recommendations.append(
            {
                "priority": "MEDIUM",
                "badge": "badge-amber",
                "strategy": "Plan Optimization & Loyalty Credit",
                "action": "Conduct a customer usage audit; provide a temporary $15/month loyalty credit for 3 months or bundle free streaming perks.",
                "expected_impact": "Relieves acute price sensitivity while preserving top-tier fiber optic relationship.",
            }
        )

    # 5. Onboarding VIP Concierge Strategy (Short Tenure)
    tenure = float(profile.get("tenure_months", 1.0))
    if tenure <= 6.0 or "Tenure Months" in driver_names:
        recommendations.append(
            {
                "priority": "CRITICAL" if tenure <= 3 else "MEDIUM",
                "badge": "badge-crimson" if tenure <= 3 else "badge-purple",
                "strategy": "VIP Onboarding Concierge Call",
                "action": "Assign a dedicated Customer Success Specialist to conduct an onboarding satisfaction call within 48 hours.",
                "expected_impact": "Crucial for navigating the vulnerable first 90 days where ~45% of first-year churn occurs.",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "priority": "LOW",
                "badge": "badge-emerald",
                "strategy": "Proactive Loyalty Rewards",
                "action": "Maintain regular scheduled customer touchpoints and offer quarterly loyalty milestone perks.",
                "expected_impact": "Nurtures customer satisfaction and maximizes customer lifetime value (CLTV).",
            }
        )

    return recommendations


def render_retention_triage_page() -> None:
    """Render Page 4: Real-Time Customer Scoring, Triage & SHAP Waterfall."""
    render_hero(
        title="Live Customer Retention Scoring & Triage",
        subtitle="Real-time frontline risk assessment, cost-utility threshold decisioning, and signed SHAP explainability.",
        badge="Frontline Operations",
    )

    # Load production model bundle and metadata
    try:
        bundle = load_model_bundle()
        meta = load_metadata()
    except Exception as exc:
        st.error(f"Failed to load production ModelBundle: {exc}")
        return

    optimal_cutoff = float(meta.get("optimal_threshold", 0.12))

    # Initialize session state for form inputs
    if "triage_profile" not in st.session_state:
        st.session_state["triage_profile"] = get_preset_profile("High Risk New Customer")

    # 1. Preset Archetype Buttons
    st.markdown("#### ⚡ Quick-Select Customer Archetypes")
    p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 0.7])

    with p_col1:
        if st.button("🚨 High Risk New Customer", use_container_width=True):
            st.session_state["triage_profile"] = get_preset_profile("High Risk New Customer")
            st.rerun()

    with p_col2:
        if st.button("🛡️ Loyal Long-Term Customer", use_container_width=True):
            st.session_state["triage_profile"] = get_preset_profile("Loyal Long-Term Customer")
            st.rerun()

    with p_col3:
        if st.button("⚡ Drifted High-Charge Customer", use_container_width=True):
            st.session_state["triage_profile"] = get_preset_profile("Drifted High-Charge Customer")
            st.rerun()

    with p_col4:
        use_fastapi = st.checkbox("FastAPI Service", value=False, help="Score via POST http://localhost:8000/predict")

    curr = st.session_state["triage_profile"]

    # 2. Customer Attributes Entry Form
    with st.expander("📝 Customer Profile & Subscription Attributes", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            st.markdown("##### 👤 Demographics & Tenure")
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                index=0 if curr.get("gender") == "Female" else 1,
            )
            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
                index=0 if curr.get("senior_citizen") == "No" else 1,
            )
            partner = st.selectbox(
                "Partner",
                ["No", "Yes"],
                index=0 if curr.get("partner") == "No" else 1,
            )
            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"],
                index=0 if curr.get("dependents") == "No" else 1,
            )
            tenure_months = st.slider(
                "Tenure Months",
                min_value=1.0,
                max_value=72.0,
                value=float(curr.get("tenure_months", 2.0)),
                step=1.0,
            )

        with f_col2:
            st.markdown("##### 🌐 Telecom & Services")
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
                index=0 if curr.get("phone_service") == "Yes" else 1,
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes"],
                index=0 if curr.get("multiple_lines") == "No" else 1,
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
                index=["Fiber optic", "DSL", "No"].index(curr.get("internet_service", "Fiber optic")),
            )
            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes"],
                index=0 if curr.get("online_security") == "No" else 1,
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes"],
                index=0 if curr.get("online_backup") == "No" else 1,
            )
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes"],
                index=0 if curr.get("device_protection") == "No" else 1,
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes"],
                index=0 if curr.get("tech_support") == "No" else 1,
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes"],
                index=0 if curr.get("streaming_tv") == "No" else 1,
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes"],
                index=0 if curr.get("streaming_movies") == "No" else 1,
            )

        with f_col3:
            st.markdown("##### 💳 Billing & Financials")
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(curr.get("contract", "Month-to-month")),
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                index=0 if curr.get("paperless_billing") == "Yes" else 1,
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ].index(curr.get("payment_method", "Electronic check")),
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=18.0,
                max_value=140.0,
                value=float(curr.get("monthly_charges", 89.50)),
                step=1.0,
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=18.0,
                max_value=9500.0,
                value=float(curr.get("total_charges", monthly_charges * tenure_months)),
                step=10.0,
            )
            cltv = st.number_input(
                "Customer Lifetime Value (CLTV)",
                min_value=500.0,
                max_value=10000.0,
                value=float(curr.get("cltv", 3200.0)),
                step=50.0,
            )

    # Active profile dictionary
    active_profile: dict[str, Any] = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "cltv": cltv,
    }

    # 3. Real-Time Scoring & SHAP Explanation
    with st.spinner("Scoring customer risk profile & calculating SHAP attributions..."):
        prob, pred, risk_tier, cutoff_applied, encoded_df = score_customer_profile(
            active_profile,
            bundle=bundle,
            use_api=use_fastapi,
        )
        explanation = explain_customer_instance(bundle.model, encoded_df)

    # 4. Results Display: Speedometer & Decisioning
    st.markdown("---")
    r_col1, r_col2 = st.columns([1, 1.25])

    with r_col1:
        st.plotly_chart(
            generate_risk_gauge(prob, threshold=optimal_cutoff),
            use_container_width=True,
        )

        # Operational Decision Banner
        if prob >= optimal_cutoff:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 5px solid #EF4444; margin-top: -10px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span class="badge badge-crimson">🚨 CHURN RISK DETECTED</span>
                        <span style="font-size: 0.82rem; color: #EF4444; font-weight: 700;">TIER: {risk_tier.upper()}</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 0.5rem 0 0.3rem 0;">
                        Retention Intervention Authorized
                    </div>
                    <div style="font-size: 0.85rem; color: #D1D5DB; line-height: 1.5;">
                        Customer churn probability (<b>{prob:.1%}</b>) exceeds the cost-optimal decision threshold (<b>{optimal_cutoff:.1%}</b>).
                        Expected penalty for unmitigated churn is <b>$200.00</b> vs <b>$20.00</b> retention incentive cost.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 5px solid #10B981; margin-top: -10px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <span class="badge badge-emerald">✅ HEALTHY / RETAINED</span>
                        <span style="font-size: 0.82rem; color: #10B981; font-weight: 700;">TIER: {risk_tier.upper()}</span>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 0.5rem 0 0.3rem 0;">
                        No Intervention Required
                    </div>
                    <div style="font-size: 0.85rem; color: #D1D5DB; line-height: 1.5;">
                        Customer churn probability (<b>{prob:.1%}</b>) is safely below the operational threshold (<b>{optimal_cutoff:.1%}</b>).
                        Account is in good standing; standard proactive loyalty engagement applies.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with r_col2:
        st.plotly_chart(
            generate_attribution_bar_chart(
                explanation.risk_drivers,
                explanation.protective_factors,
            ),
            use_container_width=True,
        )

    # 5. Frontline Action Plan & Strategy Recommendations
    st.markdown("### 📋 Prescriptive Retention Action Plan")
    st.markdown(
        "<p style='color: #9CA3AF; font-size: 0.9rem; margin-top: -8px;'>"
        "Actionable interventions dynamically matched to this customer's dominant risk drivers and contractual profile."
        "</p>",
        unsafe_allow_html=True,
    )

    recommendations = generate_retention_recommendations(active_profile, explanation.risk_drivers)

    rec_cols = st.columns(len(recommendations) if len(recommendations) <= 3 else 3)
    for idx, rec in enumerate(recommendations):
        col = rec_cols[idx % len(rec_cols)]
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="height: 100%; margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span class="badge {rec['badge']}">{rec['priority']} PRIORITY</span>
                    </div>
                    <div style="font-weight: 700; font-size: 1.0rem; color: #FFFFFF; margin-bottom: 0.4rem;">
                        {rec['strategy']}
                    </div>
                    <div style="font-size: 0.82rem; color: #D1D5DB; margin-bottom: 0.6rem; line-height: 1.5;">
                        <b>Action:</b> {rec['action']}
                    </div>
                    <div style="font-size: 0.78rem; color: #34D399; line-height: 1.4; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.5rem;">
                        <b>Impact:</b> {rec['expected_impact']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
