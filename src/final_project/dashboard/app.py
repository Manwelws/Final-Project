"""Main Streamlit application entrypoint and multi-page router for ChurnOps."""

from __future__ import annotations

import streamlit as st

from final_project.dashboard.styles import apply_custom_styles, COLORS
from final_project.dashboard.state import load_metadata
from final_project.dashboard.pages.kpi_overview import render_kpi_overview_page


def main() -> None:
    """Run the ChurnOps Streamlit dashboard application."""
    st.set_page_config(
        page_title="ChurnOps | Executive & Retraining Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 1. Apply global dark modern styles & fonts
    apply_custom_styles()

    # 2. Load system metadata
    meta = load_metadata()
    model_algo = meta.get("algorithm", "RandomForestClassifier")
    optimal_cutoff = meta.get("optimal_threshold", 0.12)
    feature_count = meta.get("feature_count", 27)

    # 3. Sidebar Brand & Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <div style="font-size: 1.8rem;">⚡</div>
                    <div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; color: #FFFFFF;">
                            ChurnOps
                        </div>
                        <div style="font-size: 0.75rem; color: #9CA3AF; letter-spacing: 0.05em; text-transform: uppercase;">
                            Production Intelligence
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="margin-bottom: 1.25rem;">
                <span class="badge badge-emerald">● SYSTEM OPERATIONAL</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_page = st.radio(
            "NAVIGATION",
            [
                "📊 Executive KPI Overview",
                "📈 Model Benchmark & Diagnostics",
                "🔄 Drift & Retraining Monitor",
                "🎯 Live Retention Scoring & Triage",
                "🧪 MLflow Experiment Hub",
            ],
            index=0,
            label_visibility="visible",
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)

        st.markdown("### ⚙️ System Status")
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #9CA3AF; line-height: 1.8;">
                <div><b>Algorithm:</b> <code style="color: #60A5FA;">{model_algo}</code></div>
                <div><b>Optimal Cutoff:</b> <code style="color: #34D399;">{optimal_cutoff:.2f}</code></div>
                <div><b>Input Features:</b> <code>{feature_count} numeric</code></div>
                <div><b>Label Lag:</b> <code>2 weeks</code></div>
                <div><b>Retraining:</b> <code>Adaptive Exponential</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top: 2rem; font-size: 0.72rem; color: #6B7280; text-align: center;">
                ChurnOps v0.1.0 • Final Project<br>DEPI ML Engineering Track
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Route to page
    if selected_page == "📊 Executive KPI Overview":
        render_kpi_overview_page()
    elif selected_page == "📈 Model Benchmark & Diagnostics":
        try:
            from final_project.dashboard.pages.model_diagnostics import render_model_diagnostics_page

            render_model_diagnostics_page()
        except ImportError:
            st.info("📈 Model Benchmark & Diagnostics page will be rendered here (Issue #25).")
    elif selected_page == "🔄 Drift & Retraining Monitor":
        try:
            from final_project.dashboard.pages.drift_monitor import render_drift_monitor_page

            render_drift_monitor_page()
        except ImportError:
            st.info("🔄 Drift & Retraining Monitor page will be rendered here (Issue #26).")
    elif selected_page == "🎯 Live Retention Scoring & Triage":
        try:
            from final_project.dashboard.pages.retention_triage import render_retention_triage_page

            render_retention_triage_page()
        except ImportError:
            st.info("🎯 Live Retention Scoring & Triage page will be rendered here (Issue #27).")
    elif selected_page == "🧪 MLflow Experiment Hub":
        try:
            from final_project.dashboard.pages.experiment_hub import render_experiment_hub_page

            render_experiment_hub_page()
        except ImportError:
            st.info("🧪 MLflow Experiment Hub page will be rendered here (Issue #28).")


if __name__ == "__main__":
    main()
