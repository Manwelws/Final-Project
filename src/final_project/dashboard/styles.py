"""Dashboard styling, theme tokens, and custom CSS design system."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# Color Palette Tokens
COLORS = {
    "bg_dark": "#0B0F19",
    "surface": "#111827",
    "surface_card": "rgba(17, 24, 39, 0.72)",
    "border": "rgba(255, 255, 255, 0.08)",
    "primary": "#3B82F6",
    "primary_gradient": "linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%)",
    "secondary": "#8B5CF6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "text_main": "#F9FAFB",
    "text_muted": "#9CA3AF",
    "text_subtle": "#6B7280",
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {COLORS['text_main']};
}}

h1, h2, h3, h4, h5, h6 {{
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}}

/* Main App Container Styling */
.stApp {{
    background: radial-gradient(circle at 15% 15%, #131B2E 0%, {COLORS['bg_dark']} 55%, #050811 100%);
    background-attachment: fixed;
}}

/* Custom Scrollbars */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: #0B0F19;
}}
::-webkit-scrollbar-thumb {{
    background: #1F2937;
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: #374151;
}}

/* Glassmorphism Cards */
.glass-card {{
    background: {COLORS['surface_card']};
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid {COLORS['border']};
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}}

.glass-card:hover {{
    transform: translateY(-2px);
    border-color: rgba(59, 130, 246, 0.25);
    box-shadow: 0 12px 36px rgba(59, 130, 246, 0.12);
}}

/* Metric Card Layout */
.metric-container {{
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}}

.metric-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.metric-label {{
    font-family: 'Outfit', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COLORS['text_muted']};
}}

.metric-value {{
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    background: linear-gradient(135deg, #FFFFFF 30%, #D1D5DB 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.metric-delta {{
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.15rem 0.55rem;
    border-radius: 9999px;
    width: fit-content;
}}

.delta-positive {{
    background: rgba(16, 185, 129, 0.15);
    color: {COLORS['success']};
    border: 1px solid rgba(16, 185, 129, 0.3);
}}

.delta-negative {{
    background: rgba(239, 68, 68, 0.15);
    color: {COLORS['danger']};
    border: 1px solid rgba(239, 68, 68, 0.3);
}}

.delta-neutral {{
    background: rgba(59, 130, 246, 0.15);
    color: {COLORS['primary']};
    border: 1px solid rgba(59, 130, 246, 0.3);
}}

.metric-subtext {{
    font-size: 0.75rem;
    color: {COLORS['text_subtle']};
    margin-top: 0.15rem;
}}

/* Hero Section */
.hero-wrapper {{
    padding: 1.5rem 0 1.25rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid {COLORS['border']};
}}

.hero-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #60A5FA;
    background: rgba(37, 99, 235, 0.14);
    border: 1px solid rgba(96, 165, 250, 0.28);
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    margin-bottom: 0.75rem;
}}

.hero-title {{
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
    background: linear-gradient(135deg, #FFFFFF 0%, #93C5FD 50%, #C4B5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.hero-subtitle {{
    font-size: 0.98rem;
    color: {COLORS['text_muted']};
    max-width: 800px;
    line-height: 1.5;
}}

/* Sidebar Customization */
[data-testid="stSidebar"] {{
    background-color: #080B12 !important;
    border-right: 1px solid {COLORS['border']};
}}

/* Streamlit Selectboxes, Inputs & Sliders */
.stSelectbox > div > div, .stTextInput > div > div > input {{
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #F9FAFB !important;
}}

.stSelectbox > div > div:focus-within, .stTextInput > div > div > input:focus {{
    border-color: {COLORS['primary']} !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}}

/* Badge Component */
.badge {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}

.badge-emerald {{ background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.35); }}
.badge-amber {{ background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.35); }}
.badge-crimson {{ background: rgba(239, 68, 68, 0.18); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.35); }}
.badge-blue {{ background: rgba(59, 130, 246, 0.18); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.35); }}
.badge-purple {{ background: rgba(139, 92, 246, 0.18); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.35); }}

</style>
"""


def apply_custom_styles() -> None:
    """Inject global CSS stylesheets and fonts into Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_plotly_layout_defaults() -> dict:
    """Return unified dark-mode layout defaults for Plotly figures."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0)",
        plot_bgcolor="rgba(17, 24, 39, 0.4)",
        font=dict(family="Inter, sans-serif", color="#D1D5DB", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor="#1F2937",
            zerolinecolor="#374151",
            showline=True,
            linecolor="#374151",
        ),
        yaxis=dict(
            gridcolor="#1F2937",
            zerolinecolor="#374151",
            showline=True,
            linecolor="#374151",
        ),
        legend=dict(
            bgcolor="rgba(17, 24, 39, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
        ),
    )


def style_plotly_figure(fig: go.Figure) -> go.Figure:
    """Apply consistent styling and dark palette to any Plotly figure."""
    fig.update_layout(**get_plotly_layout_defaults())
    return fig


def render_hero(title: str, subtitle: str, badge: str | None = None) -> None:
    """Render a modern hero header banner with title, subtitle and optional pill badge."""
    badge_html = f'<div class="hero-pill">{badge}</div>' if badge else ""
    html = f"""
    <div class="hero-wrapper">
        {badge_html}
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_variant: str = "positive",
    subtext: str | None = None,
    icon: str | None = None,
    *,
    delta_type: str | None = None,
    sublabel: str | None = None,
) -> None:
    """Render a premium glassmorphic KPI card with delta pill and subtext."""
    effective_variant = delta_type if delta_type is not None else delta_variant
    effective_subtext = sublabel if sublabel is not None else subtext

    delta_class = f"delta-{effective_variant}"
    delta_html = f'<span class="metric-delta {delta_class}">{delta}</span>' if delta else ""
    subtext_html = f'<div class="metric-subtext">{effective_subtext}</div>' if effective_subtext else ""
    icon_html = f"<span>{icon}</span>" if icon else ""

    html = f"""
    <div class="glass-card">
        <div class="metric-container">
            <div class="metric-header">
                <span class="metric-label">{label}</span>
                {icon_html}
            </div>
            <div class="metric-value">{value}</div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 0.25rem;">
                {delta_html}
            </div>
            {subtext_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

