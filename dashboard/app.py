"""
ICU Ward Patient Risk Prediction - Streamlit Real-Time Dashboard.
"""
import os
import glob
import json
import time
import yaml
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="ICU Ward Monitor | Real-time Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #00D26A;
        margin-bottom: 10px;
    }
    .bed-card-green {
        background: linear-gradient(135deg, #0d3822 0%, #164e2d 100%);
        border: 1px solid #00D26A;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        color: white;
    }
    .bed-card-amber {
        background: linear-gradient(135deg, #4d3800 0%, #664d00 100%);
        border: 1px solid #FFD600;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        color: white;
    }
    .bed-card-red {
        background: linear-gradient(135deg, #591414 0%, #801b1b 100%);
        border: 1px solid #FF334B;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        color: white;
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3)
def load_latest_features(parquet_dir: str):
    if not os.path.exists(parquet_dir):
        return pd.DataFrame()

    files = glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True)
    if not files:
        return pd.DataFrame()

    try:
        # Load recent parquet files
        recent_files = sorted(files, key=os.path.getmtime, reverse=True)[:15]
        dfs = [pd.read_parquet(f) for f in recent_files]
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)

        # Unpack Spark window struct if present
        if "window" in combined.columns and not combined.empty:
            if isinstance(combined["window"].iloc[0], dict):
                combined["window_start"] = pd.to_datetime(combined["window"].apply(lambda w: w.get("start") if isinstance(w, dict) else None))
                combined["window_end"] = pd.to_datetime(combined["window"].apply(lambda w: w.get("end") if isinstance(w, dict) else None))
            else:
                combined["window_start"] = pd.to_datetime(combined["window"])

        return combined
    except Exception as e:
        return pd.DataFrame()

def main():
    st.title("🏥 ICU Ward Real-Time Risk Prediction Pipeline")
    st.markdown("**Big Data Stream Processing & ML Inference System** | Spark Structured Streaming & Kafka KRaft")

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    parquet_features_dir = cfg["paths"]["parquet_features_dir"]
    gt_dir = cfg["paths"]["ground_truth_dir"]

    # Sidebar controls
    st.sidebar.header("Ward Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (every 3s)", value=True)
    selected_view = st.sidebar.radio("View", ["Ward Overview Grid", "Patient Drill-Down", "System Telemetry & Architecture"])

    df = load_latest_features(parquet_features_dir)

    # Header metric indicators
    total_beds = 100
    active_monitored = len(df["patient_id"].unique()) if not df.empty else 0
    high_risk_count = 0
    med_risk_count = 0

    sort_col = "window_start" if "window_start" in df.columns else ("window_end" if "window_end" in df.columns else None)

    if not df.empty and "ml_risk_level" in df.columns:
        if sort_col:
            latest_per_patient = df.sort_values(sort_col, ascending=False).groupby("patient_id").first().reset_index()
        else:
            latest_per_patient = df.groupby("patient_id").last().reset_index()

        high_risk_count = len(latest_per_patient[latest_per_patient["ml_risk_level"] == "HIGH"])
        med_risk_count = len(latest_per_patient[latest_per_patient["ml_risk_level"] == "MEDIUM"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total ICU Beds", total_beds)
    col2.metric("Active Streams Monitored", active_monitored)
    col3.metric("🚨 Critical / High Risk", high_risk_count)
    col4.metric("⚠️ Moderate Risk", med_risk_count)

    st.divider()

    if selected_view == "Ward Overview Grid":
        st.subheader("ICU Bed Matrix (Risk Sorted)")
        if df.empty:
            st.info("Waiting for Spark Streaming micro-batches to write Parquet features... Run `generator/producer.py` and `streaming/pipeline.py` to stream live.")
        else:
            if sort_col:
                latest = df.sort_values(sort_col, ascending=False).groupby("patient_id").first().reset_index()
            else:
                latest = df.groupby("patient_id").last().reset_index()

            # Sort by ml_risk_score descending
            if "ml_risk_score" in latest.columns:
                latest = latest.sort_values("ml_risk_score", ascending=False)

            cols_per_row = 5
            rows = [latest.iloc[i:i+cols_per_row] for i in range(0, len(latest), cols_per_row)]

            for row in rows:
                cols = st.columns(cols_per_row)
                for c_idx, (_, p) in enumerate(row.iterrows()):
                    with cols[c_idx]:
                        risk_level = p.get("ml_risk_level", "LOW")
                        css_class = "bed-card-green"
                        if risk_level == "HIGH":
                            css_class = "bed-card-red"
                        elif risk_level == "MEDIUM":
                            css_class = "bed-card-amber"

                        hr = p.get("hr_mean", 0.0)
                        spo2 = p.get("spo2_mean", 0.0)
                        sbp = p.get("systolic_bp", 120.0)
                        score = p.get("ml_risk_score", 0.0)

                        st.markdown(f"""
                        <div class="{css_class}">
                            <h4 style="margin:0;">{p['bed_id']} ({p['patient_id']})</h4>
                            <p style="margin:2px 0; font-size:14px;">Risk Score: <b>{score:.2f}</b></p>
                            <p style="margin:2px 0; font-size:12px;">HR: {hr:.1f} bpm | SpO2: {spo2:.1f}%</p>
                            <p style="margin:2px 0; font-size:12px;">BP: {sbp:.0f} mmHg | NEWS: {p.get('news_score', 0)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write("")

    elif selected_view == "Patient Drill-Down":
        st.subheader("Patient Vitals & Deterioration Trajectory")
        if df.empty:
            st.info("No telemetry features loaded yet.")
        else:
            patients_list = sorted(df["patient_id"].unique())
            selected_pid = st.selectbox("Select Patient", patients_list)

            if sort_col:
                p_data = df[df["patient_id"] == selected_pid].sort_values(sort_col)
            else:
                p_data = df[df["patient_id"] == selected_pid]


            if not p_data.empty:
                # 3-chart layout
                # 1. HR & SpO2
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=list(range(len(p_data))), y=p_data["hr_mean"], name="Heart Rate (bpm)", line=dict(color="#FF5722", width=2)))
                fig1.add_trace(go.Scatter(x=list(range(len(p_data))), y=p_data["spo2_mean"], name="SpO2 (%)", line=dict(color="#00BCD4", width=2), yaxis="y2"))
                fig1.update_layout(
                    title=f"Telemetry Vitals Trace - {selected_pid}",
                    yaxis=dict(title="Heart Rate (bpm)"),
                    yaxis2=dict(title="SpO2 (%)", overlaying="y", side="right", range=[60, 105]),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig1, use_container_width=True)

                # 2. Risk Score & NEWS
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=list(range(len(p_data))), y=p_data["ml_risk_score"], name="ML Risk Probability", line=dict(color="#E91E63", width=3)))
                fig2.add_trace(go.Scatter(x=list(range(len(p_data))), y=p_data["news_score"] / 10.0, name="NEWS Score (Scaled /10)", line=dict(color="#FFC107", dash="dash")))
                fig2.add_hline(y=0.70, line_dash="dot", line_color="red", annotation_text="High Risk Threshold")
                fig2.update_layout(
                    title="Real-Time Risk Prediction vs Rule-based NEWS",
                    yaxis=dict(title="Risk Probability [0, 1]", range=[0, 1.05]),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig2, use_container_width=True)

    elif selected_view == "System Telemetry & Architecture":
        st.subheader("Big Data Architecture Specifications")
        st.markdown("""
        - **Data Ingestion:** Apache Kafka 3.8.1 (KRaft Single-Broker, 6 partitions for `vitals-telemetry`, 2 for `vitals-bp-events`, 2 for `vitals-alerts`).
        - **Stream Processing:** PySpark 3.5.3 Structured Streaming (master: `local[2]`, micro-batch trigger: 5 seconds).
        - **Stateful Windowing:** 30s event-time window with 10s slide; 15s watermark.
        - **Zero Python UDFs:** Trend slopes calculated via closed-form OLS formula inside Catalyst expressions.
        - **Machine Learning:** PySpark MLlib Pipeline (`VectorAssembler` + `StandardScaler` + `LogisticRegression` / `RandomForest`).
        """)

    if auto_refresh:
        time.sleep(3)
        st.rerun()

if __name__ == "__main__":
    main()
