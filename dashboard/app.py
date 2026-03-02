"""
app.py — Streamlit monitoring dashboard for Coal Demand Forecasting.
Provides tabs for forecast view, model performance, EDA insights,
economic impact, and drift monitoring.
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Configuration ──────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8000")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Coal Demand Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #2C3E50;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Coal Demand Forecasting Dashboard</div>',
            unsafe_allow_html=True)


def _api_call(endpoint: str, method: str = "GET", data: dict = None) -> Optional[dict]:
    """Make API call with error handling."""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"API call to {endpoint} failed: {e}")
        return None


def _load_meta() -> Dict[str, Any]:
    """Load best model metadata."""
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


# ─── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

# API status
api_status = _api_call("/health")
if api_status and api_status.get("status") == "healthy":
    st.sidebar.success(f"API: Online ({api_status.get('model_name', 'N/A')})")
else:
    st.sidebar.error("API: Offline")

meta = _load_meta()
if meta:
    st.sidebar.info(f"Model: {meta.get('model_name', 'N/A')}")
    st.sidebar.info(f"MAPE: {meta.get('mape', 'N/A'):.2f}%")

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast View",
    "🏆 Model Performance",
    "🔍 EDA Insights",
    "💰 Economic Impact",
    "🔄 Drift Monitoring",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: FORECAST VIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Coal Demand Forecast")
    st.markdown("Generate forecasts by selecting a date range and parameters.")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2025, 1, 1))
        temperature = st.slider("Temperature (°C)", 10.0, 48.0, 28.0)
        coal_price = st.number_input("Coal Price (INR/tonne)", 8000, 14000, 11000)

    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 1, 15))
        power_gen = st.slider("Power Generation (MW)", 200.0, 800.0, 520.0)
        include_holidays = st.checkbox("Include holiday effects", value=True)

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecasts..."):
            # Build batch request
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            batch_requests = []
            for d in date_range:
                is_weekend = d.dayofweek >= 5
                batch_requests.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "temperature_c": temperature,
                    "is_holiday": False,
                    "is_weekend": is_weekend,
                    "coal_price_inr": coal_price,
                    "power_generation_mw": power_gen,
                })

            result = _api_call("/predict/batch", method="POST",
                               data={"requests": batch_requests})

            if result and "forecasts" in result:
                forecasts = result["forecasts"]
                df_forecast = pd.DataFrame(forecasts)

                # Line chart with confidence bands
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_forecast["forecast_date"],
                    y=df_forecast["predicted_coal_tonnes"],
                    mode="lines+markers",
                    name="Predicted",
                    line=dict(color="#E74C3C", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=df_forecast["forecast_date"],
                    y=df_forecast["upper_bound"],
                    mode="lines",
                    name="Upper Bound",
                    line=dict(color="#E74C3C", width=0),
                    showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=df_forecast["forecast_date"],
                    y=df_forecast["lower_bound"],
                    mode="lines",
                    name="Confidence Band",
                    line=dict(color="#E74C3C", width=0),
                    fill="tonexty",
                    fillcolor="rgba(231,76,60,0.15)",
                ))
                fig.update_layout(
                    title="Coal Demand Forecast",
                    xaxis_title="Date",
                    yaxis_title="Coal (tonnes)",
                    template="plotly_white",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Forecast Table")
                st.dataframe(df_forecast, use_container_width=True)
            else:
                st.error("Failed to generate forecasts. Ensure the API is running.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Performance Comparison")

    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(comparison_path):
        df_comp = pd.read_csv(comparison_path)

        # Metric cards
        best_row = df_comp.loc[df_comp["MAPE %"].idxmin()]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", best_row["Model"])
        with col2:
            st.metric("MAPE", f"{best_row['MAPE %']:.2f}%")
        with col3:
            st.metric("RMSE", f"{best_row['RMSE']:.2f}")
        with col4:
            st.metric("R²", f"{best_row['R²']:.4f}")

        # MAPE comparison bar chart
        fig_mape = px.bar(
            df_comp, x="Model", y="MAPE %",
            title="MAPE Comparison Across Models",
            color="Model",
            color_discrete_sequence=["#3498DB", "#E74C3C", "#27AE60", "#F39C12"],
        )
        fig_mape.add_hline(y=8.0, line_dash="dash", line_color="red",
                           annotation_text="Target: 8%")
        fig_mape.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_mape, use_container_width=True)

        # Full comparison table
        st.subheader("Detailed Comparison")
        st.dataframe(df_comp.style.highlight_min(subset=["MAPE %", "RMSE", "MAE"],
                                                   color="#27AE60", axis=0),
                      use_container_width=True)

        # Best model forecast plot
        forecast_path = os.path.join(REPORTS_DIR, "best_model_forecast.png")
        if os.path.exists(forecast_path):
            st.subheader("Best Model — Actual vs Predicted")
            st.image(forecast_path)
    else:
        st.warning("No model comparison data found. Run the training pipeline first.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: EDA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Exploratory Data Analysis")

    eda_plots = {
        "SIPOC Diagram": "sipoc_diagram.png",
        "Value Chain": "value_chain.png",
        "Seasonality Decomposition": "seasonality_decomposition.png",
        "Trend Analysis": "trend_analysis.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "ACF/PACF Plots": "acf_pacf.png",
        "Outlier Boxplots": "outlier_boxplots.png",
    }

    for title, filename in eda_plots.items():
        filepath = os.path.join(REPORTS_DIR, filename)
        if os.path.exists(filepath):
            st.subheader(title)
            st.image(filepath)
            st.markdown("---")
        else:
            st.info(f"{title} not found. Run EDA pipeline to generate.")

    # Data Issues Report
    st.subheader("Data Issues Report")
    st.markdown("""
    **Typical findings from the EDA pipeline:**
    - Missing values: ~5% across key columns (forward-filled)
    - Outliers: ~3% injected and capped at IQR bounds
    - Stationarity: ADF test performed on target variable
    - Skewness: Assessed and logged
    - Recommendations: Lag features, rolling statistics, calendar features applied
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: ECONOMIC IMPACT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Economic Impact Analysis")

    if meta and "economic_impact" in meta:
        impact = meta["economic_impact"]

        # Big metric
        total_saving = impact.get("total_annual_saving_inr", 0)
        st.markdown(f"### Estimated Annual Savings: **₹{total_saving:,.0f}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Holding Cost Saved (Annual)",
                       f"₹{impact.get('holding_cost_saved_annual', 0):,.0f}")
        with col2:
            st.metric("Shortage Cost Avoided (Annual)",
                       f"₹{impact.get('shortage_cost_avoided_annual', 0):,.0f}")
        with col3:
            st.metric("Shortage Events Prevented",
                       f"{impact.get('shortage_events_prevented', 0)}")

        # Breakdown bar chart
        breakdown_data = pd.DataFrame({
            "Category": ["Holding Cost Saved", "Shortage Cost Avoided"],
            "Amount (INR)": [
                impact.get("holding_cost_saved_annual", 0),
                impact.get("shortage_cost_avoided_annual", 0),
            ]
        })

        fig_econ = px.bar(
            breakdown_data, x="Category", y="Amount (INR)",
            title="Economic Impact Breakdown",
            color="Category",
            color_discrete_sequence=["#27AE60", "#E74C3C"],
        )
        fig_econ.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_econ, use_container_width=True)

        # Additional details
        st.subheader("Details")
        detail_df = pd.DataFrame({
            "Metric": [
                "Excess Inventory Days",
                "Avg Excess Inventory (tonnes)",
                "Total Shortage Risk Days",
                "Events Prevented",
            ],
            "Value": [
                str(impact.get("excess_inventory_days", 0)),
                f"{impact.get('avg_excess_tonnes', 0):.2f}",
                str(impact.get("shortage_events_total", 0)),
                str(impact.get("shortage_events_prevented", 0)),
            ]
        })
        st.dataframe(detail_df, use_container_width=True)
    else:
        st.warning("Economic impact data not available. Run the training pipeline first.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: DRIFT MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Model Drift Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Status")
        if meta:
            st.info(f"**Model:** {meta.get('model_name', 'N/A')}")
            st.info(f"**Trained:** {meta.get('trained_on', 'N/A')}")
            st.info(f"**Baseline MAPE:** {meta.get('mape', 0):.2f}%")

    with col2:
        st.subheader("Drift Check")
        if st.button("Run Drift Detection"):
            with st.spinner("Running drift detection..."):
                try:
                    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
                    from src.monitoring import run_monitoring
                    results = run_monitoring()

                    drift = results.get("drift", {})
                    perf = results.get("performance", {})

                    if drift.get("overall_drift", False):
                        st.error("⚠️ DATA DRIFT DETECTED — Consider retraining")
                    else:
                        st.success("✅ No data drift detected")

                    if perf.get("is_degraded", False):
                        st.error(f"⚠️ Model degraded — MAPE: {perf.get('current_mape', 0):.2f}%")
                    else:
                        st.success(f"✅ Model OK — MAPE: {perf.get('current_mape', 0):.2f}%")

                    # Drift details
                    if "feature_results" in drift:
                        st.subheader("Feature-Level Drift Results")
                        drift_rows = []
                        for feat, res in drift["feature_results"].items():
                            drift_rows.append({
                                "Feature": feat,
                                "KS Statistic": f"{res['ks_statistic']:.4f}",
                                "p-value": f"{res['p_value']:.6f}",
                                "Drift": "⚠️ Yes" if res["drift_detected"] else "✅ No",
                            })
                        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

                except Exception as e:
                    st.error(f"Monitoring failed: {e}")

    # Simulated MAPE trend
    st.subheader("MAPE Trend (Simulated Windows)")
    np.random.seed(42)
    baseline_mape = meta.get("mape", 5.0) if meta else 5.0
    windows = pd.date_range(start="2024-07-01", periods=12, freq="ME")
    mape_values = baseline_mape + np.random.normal(0, 1.5, len(windows))
    mape_values = np.clip(mape_values, 1, 15)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=windows, y=mape_values,
        mode="lines+markers",
        name="MAPE",
        line=dict(color="#3498DB", width=2),
    ))
    fig_trend.add_hline(y=10.0, line_dash="dash", line_color="red",
                         annotation_text="Retrain Threshold: 10%")
    fig_trend.update_layout(
        title="MAPE Over Time Windows",
        xaxis_title="Window End Date",
        yaxis_title="MAPE (%)",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7F8C8D;'>"
    "Coal Demand Forecasting — Masters in Business Analytics Foundation Project"
    "</div>",
    unsafe_allow_html=True,
)
