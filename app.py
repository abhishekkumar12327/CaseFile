"""
CASEFILE — Geospatial Investigation Support System (India)
Production-Ready Dashboard: Case Selection → Verification → Analysis Pipeline →
Movement Patterns → Probable Areas → Predicted Routes → Priority Scoring →
Model Explanations → Satellite Geospatial Map → Investigation Summary.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import json
import time

import os

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

# Page config
st.set_page_config(
    page_title="CASEFILE — Geospatial Investigation Support (India)",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Production CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        text-align: center;
        letter-spacing: -0.5px;
        margin-top: -0.5rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #475569;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .stepper-container {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        color: #f8fafc;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-around;
        align-items: center;
        font-size: 0.82rem;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stepper-item {
        color: #94a3b8;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .stepper-item.active {
        color: #38bdf8;
        font-weight: 700;
    }
    .priority-very-high { color: #dc2626; font-weight: 700; }
    .priority-high { color: #ea580c; font-weight: 700; }
    .priority-medium { color: #d97706; font-weight: 600; }
    .priority-low { color: #16a34a; font-weight: 600; }
    .disclaimer-banner {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        color: #1e40af;
        padding: 0.75rem 1rem;
        margin: 1rem 0;
        border-radius: 6px;
        font-size: 0.85rem;
        line-height: 1.4;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────────

def check_api_health():
    """Check backend status quietly."""
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=3)
        return resp.status_code == 200, resp.json()
    except requests.exceptions.ConnectionError:
        return False, {}


def get_priority_color(level):
    colors = {
        "Very High": "#dc2626",
        "High": "#ea580c",
        "Medium": "#d97706",
        "Low": "#16a34a",
    }
    return colors.get(level, "#64748b")


INDIAN_AREA_OPTIONS = [
    "Connaught Place Hub",
    "Cyber City Gurgaon",
    "Noida Sector 18",
    "Hauz Khas Village",
    "Dwarka Sector 21",
    "Aerocity Hub",
]

AGE_OPTIONS = [
    "Child (5-12)", "Teen (13-17)", "Young Adult (18-35)",
    "Adult (36-60)", "Senior (60+)",
]

WEATHER_OPTIONS = ["Clear", "Rainy", "Foggy", "Overcast", "Windy"]
DAY_OPTIONS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ─── Session State Initialization ──────────────────────────────────────

if "stage" not in st.session_state:
    st.session_state.stage = "case_selection"
if "selected_case" not in st.session_state:
    st.session_state.selected_case = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "selected_area_idx" not in st.session_state:
    st.session_state.selected_area_idx = 0


# ─── Clean Production Sidebar (No Clutter) ───────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 CASEFILE")
    st.caption("Geospatial Investigation Support System")
    st.markdown("---")

    api_ok, _ = check_api_health()

    st.markdown("### Navigation")
    stages = [
        ("case_selection", "📋 Case Selection"),
        ("case_info", "📝 Case Details"),
        ("case_overview", "📊 Overview & Run"),
        ("results", "📈 Analysis Dashboard"),
    ]
    for stage_key, stage_label in stages:
        if st.button(stage_label, key=f"nav_{stage_key}", use_container_width=True):
            if stage_key == "results" and st.session_state.analysis_results is None:
                st.warning("Run analysis first")
            else:
                st.session_state.stage = stage_key
                st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer-banner">'
        "⚠️ <strong>Synthetic Simulation</strong><br>"
        "All cases and movement paths are privacy-compliant synthetic data for investigation support."
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Top Header & Workflow Progress Stepper ────────────────────────────

st.markdown('<div class="main-header">CASEFILE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Geospatial Movement Analysis & Priority Estimation System</div>',
    unsafe_allow_html=True,
)

# Workflow Stepper Header
st.markdown("""
<div class="stepper-container">
    <div class="stepper-item active"><span>1. Data Acquisition</span> ➔</div>
    <div class="stepper-item active"><span>2. Validation</span> ➔</div>
    <div class="stepper-item active"><span>3. Feature Engineering</span> ➔</div>
    <div class="stepper-item active"><span>4. Movement Patterns</span> ➔</div>
    <div class="stepper-item active"><span>5. XGBoost Prediction</span> ➔</div>
    <div class="stepper-item active"><span>6. Priority Scoring</span> ➔</div>
    <div class="stepper-item active"><span>7. Explanation</span> ➔</div>
    <div class="stepper-item active"><span>8. Satellite Map</span></div>
</div>
""", unsafe_allow_html=True)


# ─── STAGE 1: Case Selection / Creation ───────────────────────────────

if st.session_state.stage == "case_selection":
    tab_select, tab_create = st.tabs(["📁 Select Fictional Case", "➕ Create Fictional Case"])

    with tab_select:
        st.subheader("Select an Existing Fictional Scenario")
        if not api_ok:
            st.error("Backend server unavailable. Please launch backend with `uvicorn backend.main:app`.")
        else:
            try:
                resp = requests.get(f"{API_BASE}/api/cases", timeout=5)
                cases = resp.json().get("cases", [])
                if not cases:
                    st.info("No cases found. Create a new case.")
                else:
                    case_ids = [c["Case_ID"] for c in cases]
                    selected_id = st.selectbox("Choose a case scenario:", case_ids)
                    selected = next(c for c in cases if c["Case_ID"] == selected_id)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Case ID", selected["Case_ID"])
                        st.metric("Age Group", selected["Age_Group"])
                    with col2:
                        st.metric("Last Coordinates", f"{selected['Last_Latitude']:.4f}, {selected['Last_Longitude']:.4f}")
                        st.metric("Usual Area", selected["Usual_Area"])
                    with col3:
                        st.metric("Last Seen", selected["Last_Seen_Time"])
                        st.metric("Time Since Last Seen", f"{selected['Time_Since_Last_Seen']}h")

                    if st.button("📋 Select This Case", type="primary", use_container_width=True):
                        st.session_state.selected_case = selected
                        st.session_state.stage = "case_info"
                        st.rerun()
            except Exception as e:
                st.error(f"Failed to load cases: {e}")

    with tab_create:
        st.subheader("Create a New Synthetic Investigation Scenario (India Region)")
        if not api_ok:
            st.error("Backend server unavailable.")
        else:
            with st.form("create_case_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age_group = st.selectbox("Age Group", AGE_OPTIONS, index=3)
                    last_lat = st.number_input("Last Known Latitude (India)", value=28.631500, min_value=8.0, max_value=37.0, format="%.6f")
                    last_lon = st.number_input("Last Known Longitude (India)", value=77.216700, min_value=68.0, max_value=97.0, format="%.6f")
                    last_seen = st.text_input("Last Seen Timestamp (YYYY-MM-DD HH:MM:SS)", value="2026-09-16 18:00:00")
                    day = st.selectbox("Day of Week", DAY_OPTIONS, index=2)

                with col2:
                    weather = st.selectbox("Weather Condition", WEATHER_OPTIONS)
                    usual_area = st.selectbox("Usual Area", INDIAN_AREA_OPTIONS, index=0)
                    prev_area = st.selectbox("Previous Known Area", INDIAN_AREA_OPTIONS, index=1)
                    avg_dist = st.number_input("Average Distance (km)", value=8.5, min_value=0.0)
                    avg_speed = st.number_input("Average Speed (km/h)", value=16.0, min_value=0.0)

                time_since = st.slider("Hours Since Last Seen", 0.5, 72.0, 12.0, 0.5)

                submitted = st.form_submit_button("➕ Create Case Scenario", type="primary", use_container_width=True)

                if submitted:
                    case_payload = {
                        "Age_Group": age_group,
                        "Last_Latitude": last_lat,
                        "Last_Longitude": last_lon,
                        "Last_Seen_Time": last_seen,
                        "Day": day,
                        "Weather": weather,
                        "Usual_Area": usual_area,
                        "Previous_Area": prev_area,
                        "Average_Distance": avg_dist,
                        "Average_Speed": avg_speed,
                        "Time_Since_Last_Seen": time_since,
                    }
                    try:
                        resp = requests.post(f"{API_BASE}/api/cases", json=case_payload, timeout=10)
                        if resp.status_code == 200:
                            created = resp.json()
                            st.session_state.selected_case = created
                            st.session_state.stage = "case_info"
                            st.success(f"Case {created['Case_ID']} created successfully!")
                            st.rerun()
                        else:
                            error_detail = resp.json().get("detail", resp.text)
                            st.error(f"Validation error: {error_detail}")
                    except Exception as e:
                        st.error(f"Failed to create case: {e}")


# ─── STAGE 2: Case Information ────────────────────────────────────────

elif st.session_state.stage == "case_info":
    st.markdown("## 📝 Case Scenario Details")

    case = st.session_state.selected_case
    if case is None:
        st.warning("No case selected.")
        if st.button("← Back to Selection"):
            st.session_state.stage = "case_selection"
            st.rerun()
    else:
        st.markdown("Review the input scenario metadata before triggering the analysis pipeline.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Identifiers")
            st.write(f"**Case ID:** `{case.get('Case_ID', 'N/A')}`")
            st.write(f"**Person ID:** `{case.get('Person_ID', 'N/A')}`")
            st.write(f"**Age Group:** {case.get('Age_Group', 'N/A')}")

        with col2:
            st.markdown("### Location & Time")
            st.write(f"**Last Latitude:** `{case.get('Last_Latitude', 'N/A')}`")
            st.write(f"**Last Longitude:** `{case.get('Last_Longitude', 'N/A')}`")
            st.write(f"**Last Seen:** {case.get('Last_Seen_Time', 'N/A')}")
            st.write(f"**Day:** {case.get('Day', 'N/A')}")

        with col3:
            st.markdown("### Movement Profile")
            st.write(f"**Usual Area:** {case.get('Usual_Area', 'N/A')}")
            st.write(f"**Previous Area:** {case.get('Previous_Area', 'N/A')}")
            st.write(f"**Avg Distance:** {case.get('Average_Distance', 'N/A')} km")
            st.write(f"**Avg Speed:** {case.get('Average_Speed', 'N/A')} km/h")
            st.write(f"**Time Elapsed:** {case.get('Time_Since_Last_Seen', 'N/A')} hours")

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back to Selection", use_container_width=True):
                st.session_state.stage = "case_selection"
                st.rerun()
        with col_next:
            if st.button("Proceed to Overview & Run →", type="primary", use_container_width=True):
                st.session_state.stage = "case_overview"
                st.rerun()


# ─── STAGE 3: Case Overview & Analysis Trigger ─────────────────────────

elif st.session_state.stage == "case_overview":
    st.markdown("## 📊 Case Overview & Pipeline Execution")

    case = st.session_state.selected_case
    if case is None:
        st.warning("No case selected.")
        if st.button("← Back"):
            st.session_state.stage = "case_selection"
            st.rerun()
    else:
        st.markdown("Confirm case details and execute the backend investigation-support pipeline.")

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Case ID", case.get("Case_ID", "N/A"))
        with col2:
            st.metric("Age Group", case.get("Age_Group", "N/A"))
        with col3:
            st.metric("Time Elapsed", str(case.get("Time_Since_Last_Seen", "N/A")) + "h ago")
        with col4:
            st.metric("Weather", case.get("Weather", "N/A"))

        col5, col6 = st.columns(2)
        with col5:
            st.metric("Last Coordinates", f"{case.get('Last_Latitude', 0):.6f}, {case.get('Last_Longitude', 0):.6f}")
        with col6:
            st.metric("Usual Area", case.get("Usual_Area", "N/A"))

        st.markdown("---")

        col_back, col_run = st.columns([1, 2])
        with col_back:
            if st.button("← Back to Details", use_container_width=True):
                st.session_state.stage = "case_info"
                st.rerun()
        with col_run:
            if st.button("🚀 Run Geospatial Analysis Pipeline", type="primary", use_container_width=True):
                if not api_ok:
                    st.error("Backend server is unavailable. Start FastAPI server first.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    steps = [
                        (10, "Validating case schema..."),
                        (25, "Processing movement features..."),
                        (40, "Running spatial clustering..."),
                        (55, "Detecting anomalous movement points..."),
                        (70, "Executing XGBoost location prediction..."),
                        (80, "Computing Markov route matrix..."),
                        (90, "Calculating priority scores..."),
                        (95, "Generating model explanations..."),
                    ]

                    for progress, msg in steps:
                        progress_bar.progress(progress)
                        status_text.markdown(f"**{msg}**")
                        time.sleep(0.2)

                    try:
                        resp = requests.post(
                            f"{API_BASE}/api/analyze",
                            json=case,
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            progress_bar.progress(100)
                            status_text.markdown("**✅ Analysis complete!**")
                            st.session_state.analysis_results = resp.json()
                            st.session_state.stage = "results"
                            time.sleep(0.4)
                            st.rerun()
                        else:
                            progress_bar.empty()
                            error_detail = resp.json().get("detail", resp.text)
                            status_text.error(f"Analysis failed: {error_detail}")
                    except Exception as e:
                        progress_bar.empty()
                        status_text.error(f"Analysis error: {e}")


# ─── STAGE 4: Analysis Results Dashboard ───────────────────────────────

elif st.session_state.stage == "results":
    results = st.session_state.analysis_results
    case = st.session_state.selected_case

    if results is None:
        st.warning("No analysis results. Run analysis first.")
        if st.button("← Back"):
            st.session_state.stage = "case_overview"
            st.rerun()
    else:
        st.markdown(f"## 📈 Investigation Dashboard — Case {results.get('case_id', 'N/A')}")
        st.markdown("---")

        tab_movement, tab_areas, tab_routes, tab_priority, tab_explain, tab_map, tab_summary = st.tabs([
            "🏃 Movement Patterns",
            "📍 Probable Areas",
            "🛤️ Predicted Routes",
            "⚡ Priority Scoring",
            "💡 Model Explanation",
            "🗺️ India Satellite Map",
            "📋 Investigation Summary",
        ])

        # ── 1. Movement Patterns Tab ──
        with tab_movement:
            st.markdown("### Movement Patterns & Anomaly Detection")

            st.markdown("#### Top Visited Grid Locations")
            freq_locs = results.get("frequent_locations", [])
            if freq_locs:
                freq_df = pd.DataFrame(freq_locs)
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = px.bar(
                        freq_df.head(10),
                        x="grid_cell_id", y="visit_count",
                        title="Top Visited Spatial Locations",
                        color="visit_count",
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(xaxis_tickangle=-45, height=380)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(freq_df[["grid_cell_id", "visit_count", "avg_speed"]].head(10), hide_index=True, use_container_width=True)
            else:
                st.info("No frequent location data available.")

            st.markdown("#### Trajectory Speed & Distance Distributions")
            patterns = results.get("movement_patterns", [])
            if patterns:
                pat_df = pd.DataFrame(patterns)
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.box(pat_df, y="avg_speed_kmh", title="Speed Distribution (km/h)", color_discrete_sequence=["#2563eb"])
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.box(pat_df, y="total_distance_m", title="Distance Distribution (m)", color_discrete_sequence=["#0284c7"])
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Detected Movement Anomalies")
            st.markdown(
                '<div class="disclaimer-banner">'
                "An anomaly is a statistical outlier in movement speed or distance. "
                "It is <strong>not</strong> proof of wrongdoing or suspicious activity."
                "</div>",
                unsafe_allow_html=True,
            )
            anomalies = results.get("anomalies", [])
            if anomalies:
                anom_df = pd.DataFrame(anomalies)
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = px.scatter(
                        anom_df, x="longitude", y="latitude",
                        color="anomaly_score", size="speed",
                        title="Anomalous Movement Points",
                        color_continuous_scale="Reds",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.metric("Total Anomalies Detected", len(anomalies))
                    anom_idx = st.selectbox(
                        "Inspect Anomaly Point:",
                        range(len(anomalies)),
                        format_func=lambda i: f"Point #{i+1} ({anomalies[i]['latitude']:.4f}, {anomalies[i]['longitude']:.4f})",
                    )
                    selected_anom = anomalies[anom_idx]
                    st.write(f"**Coordinates:** `{selected_anom['latitude']:.6f}, {selected_anom['longitude']:.6f}`")
                    st.write(f"**Anomaly Score:** `{selected_anom['anomaly_score']:.4f}`")
                    st.write(f"**Speed:** `{selected_anom['speed']:.1f} km/h`")
                    st.write(f"**Step Distance:** `{selected_anom['distance']:.1f} m`")

        # ── 2. Probable Areas Tab ──
        with tab_areas:
            st.markdown("### Probable Areas (XGBoost Prediction)")
            st.markdown(
                '<div class="disclaimer-banner">'
                "Probabilities reflect statistical model estimates based on feature affinity, not confirmed facts."
                "</div>",
                unsafe_allow_html=True,
            )

            areas = results.get("probable_areas", [])
            if areas:
                selected_idx = st.selectbox(
                    "Select an area to focus:",
                    range(len(areas)),
                    format_func=lambda i: f"{areas[i]['area_name']} — {areas[i]['priority_level']} Priority ({areas[i]['priority_score']} pts)",
                    key="area_selector",
                )
                st.session_state.selected_area_idx = selected_idx
                area = areas[selected_idx]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Target Area", area["area_name"])
                with col2:
                    st.metric("ML Probability", f"{area['ml_probability']:.1%}")
                with col3:
                    st.metric("Priority Score", f"{area['priority_score']} / 100")
                with col4:
                    color = get_priority_color(area["priority_level"])
                    st.markdown(
                        f'<div style="text-align:center;padding-top:0.4rem;">'
                        f'<span style="font-size:0.8rem;color:#64748b;">Priority Band</span><br>'
                        f'<span style="font-size:1.6rem;font-weight:800;color:{color};">'
                        f'{area["priority_level"]}</span></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("#### All Destination Area Rankings")
                areas_df = pd.DataFrame(areas)
                display_cols = ["area_name", "ml_probability", "priority_score", "priority_level"]
                st.dataframe(
                    areas_df[display_cols].style.format({
                        "ml_probability": "{:.2%}",
                        "priority_score": "{:.1f}",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

        # ── 3. Predicted Routes Tab ──
        with tab_routes:
            st.markdown("### Predicted Routes (Markov Transition Matrix)")
            st.markdown("Route paths predicted using step-by-step transition probabilities.")

            routes = results.get("predicted_routes", [])
            if routes:
                for i, route in enumerate(routes):
                    path = route.get("path", [])
                    prob = route.get("probability", 0)
                    with st.expander(f"Route Path #{i+1} — Probability: {prob:.4f}", expanded=(i == 0)):
                        st.write(" ➔ ".join(path))
                        st.progress(min(prob * 100, 100) / 100)
            else:
                st.info("No route paths predicted.")

        # ── 4. Priority Scoring Tab ──
        with tab_priority:
            st.markdown("### Priority Scoring Engine")
            st.markdown("Weighted multi-factor score computation:")

            weights_data = {
                "Factor": ["ML Probability", "Historical Visit Frequency", "Route Similarity", "Distance Relevance", "Time Relevance", "Anomaly Evidence"],
                "Formula Weight": ["30%", "20%", "15%", "15%", "10%", "10%"],
            }
            st.table(pd.DataFrame(weights_data))

            areas = results.get("probable_areas", [])
            if areas:
                areas_df = pd.DataFrame(areas)
                fig = px.bar(
                    areas_df, x="area_name", y="priority_score",
                    color="priority_level",
                    color_discrete_map={
                        "Very High": "#dc2626", "High": "#ea580c",
                        "Medium": "#d97706", "Low": "#16a34a",
                    },
                    title="Priority Score Comparison",
                )
                fig.update_layout(xaxis_tickangle=-25, height=380)
                st.plotly_chart(fig, use_container_width=True)

                idx = st.session_state.selected_area_idx
                if idx < len(areas):
                    area = areas[idx]
                    st.markdown(f"#### Score Component Contribution: {area['area_name']}")
                    breakdown = {
                        "Factor": ["ML Probability", "Visit Frequency", "Route Similarity", "Distance Relevance", "Time Relevance", "Anomaly Evidence"],
                        "Raw Value": [area["ml_probability"], area["visit_frequency"], area["route_similarity"], area["distance_relevance"], area["time_relevance"], area["anomaly_evidence"]],
                        "Weight": [0.30, 0.20, 0.15, 0.15, 0.10, 0.10],
                    }
                    bd_df = pd.DataFrame(breakdown)
                    bd_df["Weighted Score"] = bd_df["Raw Value"] * bd_df["Weight"] * 100
                    fig2 = px.bar(bd_df, x="Factor", y="Weighted Score", title="Factor Contributions", color="Factor")
                    fig2.update_layout(showlegend=False, height=320)
                    st.plotly_chart(fig2, use_container_width=True)

        # ── 5. Model Explanation Tab ──
        with tab_explain:
            st.markdown("### Model Explanation & Reasoning")

            explanations = results.get("explanations", [])
            if explanations:
                exp_idx = st.selectbox(
                    "Select area to explain:",
                    range(len(explanations)),
                    format_func=lambda i: f"{explanations[i]['area_name']} ({explanations[i]['priority_level']} Priority)",
                )
                exp = explanations[exp_idx]

                st.markdown(f"#### {exp['area_name']}")
                st.info(exp.get("summary", ""))

                st.markdown("**Supporting Investigation Factors:**")
                for factor in exp.get("supporting_factors", []):
                    icon = "🟢" if factor["contribution"] == "strong" else "🟡" if factor["contribution"] == "moderate" else "⚪"
                    st.markdown(f"{icon} **{factor['factor']}** (Weight: {factor['weight']}) — *{factor['detail']}*")

                st.markdown("**Top XGBoost Model Features:**")
                features = exp.get("top_model_features", [])
                if features:
                    feat_df = pd.DataFrame(features)
                    fig = px.bar(feat_df, x="feature", y="importance", title="Feature Importance Breakdown", color="importance", color_continuous_scale="Viridis")
                    fig.update_layout(xaxis_tickangle=-25, height=320)
                    st.plotly_chart(fig, use_container_width=True)

        # ── 6. India Satellite Map Tab ──
        with tab_map:
            st.markdown("### 🗺️ India Satellite Geospatial Map")
            st.markdown("Explore Indian locations, predicted target areas, movement anomalies, and route corridors on satellite imagery.")

            case = st.session_state.selected_case or {}
            areas = results.get("probable_areas", [])
            anomalies = results.get("anomalies", [])
            routes = results.get("predicted_routes", [])
            freq_locs = results.get("frequent_locations", [])

            # Default map focus centered on India / Delhi-NCR
            idx = st.session_state.get("selected_area_idx", 0)
            if areas and idx < len(areas):
                center_lat = float(areas[idx].get("area_latitude", case.get("Last_Latitude", 28.6139)))
                center_lon = float(areas[idx].get("area_longitude", case.get("Last_Longitude", 77.2090)))
                zoom_lvl = 13
            else:
                center_lat = float(case.get("Last_Latitude", 28.6139))
                center_lon = float(case.get("Last_Longitude", 77.2090))
                zoom_lvl = 12

            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                if areas:
                    focus_area_idx = st.selectbox(
                        "Map Focus Area:",
                        range(len(areas)),
                        index=min(idx, len(areas) - 1),
                        format_func=lambda i: f"{areas[i]['area_name']} ({areas[i]['priority_level']} Priority)",
                        key="map_focus_area_selector",
                    )
                    st.session_state.selected_area_idx = focus_area_idx
                    center_lat = float(areas[focus_area_idx]["area_latitude"])
                    center_lon = float(areas[focus_area_idx]["area_longitude"])

            with col_ctrl2:
                selected_route_idx = 0
                if routes:
                    selected_route_idx = st.selectbox(
                        "Highlight Route Corridor:",
                        range(len(routes)),
                        format_func=lambda i: f"Route #{i+1} (Probability: {routes[i]['probability']:.4f})",
                        key="map_route_selector",
                    )

            # Esri World Imagery Satellite Tile Layer
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_lvl,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
            )

            # Add OpenStreetMap labels overlay on top of satellite imagery
            folium.TileLayer(
                tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attr="OpenStreetMap",
                name="Street Overlay",
                overlay=True,
                opacity=0.4,
            ).add_to(m)

            # Last Known Location (Red Pin)
            last_lat_val = float(case.get("Last_Latitude", 28.6315))
            last_lon_val = float(case.get("Last_Longitude", 77.2167))
            folium.Marker(
                [last_lat_val, last_lon_val],
                popup=f"<b>Last Known Location</b><br>{last_lat_val:.6f}, {last_lon_val:.6f}",
                tooltip="Last Known Location",
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(m)

            # Top Frequent Locations (Blue Circle Markers)
            for loc in freq_locs[:10]:
                folium.CircleMarker(
                    [loc["avg_lat"], loc["avg_lon"]],
                    radius=max(4, min(loc["visit_count"] / 4, 14)),
                    color="#2563eb", fill=True, fill_color="#3b82f6", fill_opacity=0.6,
                    popup=f"<b>Frequent Hub: {loc['grid_cell_id']}</b><br>Visits: {loc['visit_count']}",
                    tooltip=f"Frequent Hub: {loc['grid_cell_id']}",
                ).add_to(m)

            # Anomalous Movement Points (Orange Circle Markers)
            for a_i, anom in enumerate(anomalies[:15]):
                folium.CircleMarker(
                    [anom["latitude"], anom["longitude"]],
                    radius=6, color="#ea580c", fill=True, fill_color="#f97316", fill_opacity=0.8,
                    popup=f"<b>Anomaly #{a_i+1}</b><br>Score: {anom['anomaly_score']:.3f}<br>Speed: {anom['speed']:.1f} km/h",
                    tooltip=f"Anomaly #{a_i+1}",
                ).add_to(m)

            # Probable Areas (Color-Coded Flag Markers)
            area_colors = {"Very High": "darkred", "High": "red", "Medium": "orange", "Low": "green"}
            for a_i, area in enumerate(areas):
                color = area_colors.get(area["priority_level"], "gray")
                is_selected = (a_i == st.session_state.selected_area_idx)
                icon_type = "star" if is_selected else "flag"
                folium.Marker(
                    [area["area_latitude"], area["area_longitude"]],
                    popup=(f"<b>{area['area_name']}</b><br>"
                           f"Priority: {area['priority_level']} ({area['priority_score']} pts)<br>"
                           f"Probability: {area['ml_probability']:.1%}"),
                    tooltip=f"{'[Selected Area] ' if is_selected else ''}{area['area_name']} ({area['priority_level']})",
                    icon=folium.Icon(color=color, icon=icon_type),
                ).add_to(m)

            # Highlighted Route Polyline
            if routes and freq_locs:
                grid_coords = {loc["grid_cell_id"]: (loc["avg_lat"], loc["avg_lon"]) for loc in freq_locs}
                for i, route in enumerate(routes):
                    path = route.get("path", [])
                    coords = [grid_coords[cell] for cell in path if cell in grid_coords]

                    if len(coords) >= 2:
                        is_route_selected = (i == selected_route_idx)
                        color = "#a855f7" if is_route_selected else "#94a3b8"
                        weight = 5 if is_route_selected else 2
                        opacity = 0.95 if is_route_selected else 0.4

                        folium.PolyLine(
                            coords, color=color,
                            weight=weight, opacity=opacity,
                            popup=f"Route Corridor #{i+1} (Prob: {route['probability']:.4f})",
                            tooltip=f"Route Corridor #{i+1} {'(Highlighted)' if is_route_selected else ''}",
                        ).add_to(m)

            st_folium(m, width=None, height=580, use_container_width=True)

        # ── 7. Investigation Summary Tab ──
        with tab_summary:
            st.markdown("### Investigation-Support Consolidated Summary")
            st.markdown(
                '<div class="disclaimer-banner">'
                "This report presents <strong>probabilistic investigation support estimates</strong>. "
                "All predictions require human investigative judgment."
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown(results.get("summary", "No summary available."))
            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frequent Hubs", len(results.get("frequent_locations", [])))
            with col2:
                st.metric("Anomalies Flagged", len(results.get("anomalies", [])))
            with col3:
                st.metric("Probable Areas", len(results.get("probable_areas", [])))
            with col4:
                st.metric("Predicted Routes", len(results.get("predicted_routes", [])))

            areas = results.get("probable_areas", [])
            if areas:
                st.markdown("#### Ranked Probable Destination Areas")
                for i, area in enumerate(areas[:5]):
                    color = get_priority_color(area["priority_level"])
                    st.markdown(
                        f"**{i+1}. {area['area_name']}** — "
                        f'<span style="color:{color};font-weight:700;">'
                        f"{area['priority_level']} Priority</span> "
                        f"(Score: {area['priority_score']} pts, "
                        f"Probability: {area['ml_probability']:.1%})",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            if st.button("🔄 Analyze Another Scenario", type="primary", use_container_width=True):
                st.session_state.analysis_results = None
                st.session_state.selected_case = None
                st.session_state.stage = "case_selection"
                st.rerun()
