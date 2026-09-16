"""
FastAPI backend for CASEFILE.
Orchestrates the analysis pipeline: data processing, ML models, priority scoring,
and explanation generation. Serves structured results to the Streamlit frontend.
"""

import os
import sys
import uuid
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from backend.schemas import (
    CaseInput, AnalysisResponse, CaseRecord, ErrorResponse,
    FrequentLocation, AnomalyPoint, ProbableArea, RouteStep,
    AreaExplanation, ExplanationFactor, FeatureImportance,
    MovementPattern, ClusterInfo,
)
from src.ml_pipeline import (
    predict_location, predict_routes, predict_clusters,
    detect_anomalies, get_frequent_locations,
)
from src.priority_scoring import score_predicted_areas
from src.explanation import explain_all_areas

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CASES_PATH = os.path.join(DATA_DIR, "synthetic", "fictional_cases.csv")

app = FastAPI(
    title="CASEFILE API",
    description="ML-powered investigation-support system backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    models_exist = os.path.exists(os.path.join(models_dir, "location_model.joblib"))
    return {
        "status": "healthy",
        "models_loaded": models_exist,
        "data_available": os.path.exists(CASES_PATH),
    }


@app.get("/api/cases")
def list_cases():
    """List all available fictional cases."""
    if not os.path.exists(CASES_PATH):
        return {"cases": []}

    df = pd.read_csv(CASES_PATH).fillna("")
    cases = df.to_dict(orient="records")
    return {"cases": cases}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    """Get a specific case by ID."""
    if not os.path.exists(CASES_PATH):
        raise HTTPException(status_code=404, detail="Case data not found")

    df = pd.read_csv(CASES_PATH).fillna("")
    case_row = df[df["Case_ID"] == case_id]

    if case_row.empty:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    return {"case": case_row.iloc[0].to_dict()}


@app.post("/api/cases", response_model=CaseRecord)
def create_case(case_input: CaseInput):
    """Create a new fictional case and save it."""
    case_id = case_input.Case_ID or f"CASE_{uuid.uuid4().hex[:8].upper()}"
    person_id = case_input.Person_ID or f"PERSON_{uuid.uuid4().hex[:8].upper()}"

    new_case = {
        "Case_ID": case_id,
        "Person_ID": person_id,
        "Age_Group": case_input.Age_Group,
        "Last_Latitude": case_input.Last_Latitude,
        "Last_Longitude": case_input.Last_Longitude,
        "Last_Seen_Time": case_input.Last_Seen_Time,
        "Day": case_input.Day,
        "Weather": case_input.Weather,
        "Usual_Area": case_input.Usual_Area,
        "Average_Distance": case_input.Average_Distance,
        "Average_Speed": case_input.Average_Speed,
        "Previous_Area": case_input.Previous_Area,
        "Time_Since_Last_Seen": case_input.Time_Since_Last_Seen,
        "Target_Area": "",
    }

    if os.path.exists(CASES_PATH):
        df = pd.read_csv(CASES_PATH)
        new_df = pd.DataFrame([new_case])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame([new_case])

    os.makedirs(os.path.dirname(CASES_PATH), exist_ok=True)
    df.to_csv(CASES_PATH, index=False)

    return CaseRecord(**new_case)


@app.post("/api/analyze")
def run_analysis(case_input: CaseInput):
    """
    Run the complete analysis pipeline for a case.
    Steps: clustering → anomaly detection → location prediction →
           route prediction → priority scoring → explanation.
    """
    case_data = case_input.model_dump()
    case_id = case_data.get("Case_ID") or f"CASE_{uuid.uuid4().hex[:8].upper()}"

    # Validate models exist
    models_dir = os.path.join(PROJECT_ROOT, "models")
    required_models = [
        "location_model.joblib", "kmeans_model.joblib",
        "isolation_forest.joblib", "transition_matrix.joblib",
    ]
    for model_file in required_models:
        if not os.path.exists(os.path.join(models_dir, model_file)):
            raise HTTPException(
                status_code=500,
                detail=f"Model {model_file} not found. Run train_models.py first.",
            )

    # Load movement data for context
    movement_path = os.path.join(DATA_DIR, "processed", "movement_features.csv")
    if not os.path.exists(movement_path):
        raise HTTPException(status_code=500, detail="Movement features data not found")

    movement_df = pd.read_csv(movement_path)

    # 1. Frequent Locations
    freq_locations = get_frequent_locations(top_n=10)

    # 2. Movement Patterns (trajectory-level summaries)
    pattern_cols = ["user_id", "total_distance_m", "avg_speed_kmh",
                    "max_speed_kmh", "duration_minutes", "locations_visited_count"]
    available_cols = [c for c in pattern_cols if c in movement_df.columns]
    patterns_df = movement_df[available_cols].drop_duplicates(
        subset=["user_id"]
    ).head(10)
    movement_patterns = []
    for _, row in patterns_df.iterrows():
        movement_patterns.append(MovementPattern(
            user_id=str(row.get("user_id", "")),
            total_distance_m=float(row.get("total_distance_m", 0)),
            avg_speed_kmh=float(row.get("avg_speed_kmh", 0)),
            max_speed_kmh=float(row.get("max_speed_kmh", 0)),
            duration_minutes=float(row.get("duration_minutes", 0)),
            locations_visited_count=int(row.get("locations_visited_count", 0)),
        ))

    # 3. Clustering
    sample_points = movement_df.head(500).copy()
    cluster_labels = predict_clusters(sample_points)
    sample_points["cluster"] = cluster_labels

    clusters = []
    for cid in sorted(sample_points["cluster"].unique()):
        cluster_pts = sample_points[sample_points["cluster"] == cid]
        clusters.append(ClusterInfo(
            cluster_id=int(cid),
            center_lat=float(cluster_pts["latitude"].mean()),
            center_lon=float(cluster_pts["longitude"].mean()),
            point_count=len(cluster_pts),
            avg_speed=float(cluster_pts["instant_speed_kmh"].mean()),
        ))

    # 4. Anomaly Detection
    anomaly_sample = movement_df.sample(min(500, len(movement_df)), random_state=42).copy()
    anomaly_labels, anomaly_scores = detect_anomalies(anomaly_sample)
    anomaly_sample["anomaly_label"] = anomaly_labels
    anomaly_sample["anomaly_score"] = anomaly_scores

    anomaly_points = []
    anomalous = anomaly_sample[anomaly_sample["anomaly_label"] == -1]
    for _, row in anomalous.head(20).iterrows():
        anomaly_points.append(AnomalyPoint(
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            anomaly_score=float(row["anomaly_score"]),
            speed=float(row.get("instant_speed_kmh", 0)),
            distance=float(row.get("step_distance_m", 0)),
        ))

    # 5. Location Prediction
    predicted_areas_raw = predict_location(case_data)

    # 6. Route Prediction
    # Find nearest grid cell to last known location
    last_lat = case_data["Last_Latitude"]
    last_lon = case_data["Last_Longitude"]
    grid_lat = round(last_lat, 2)
    grid_lon = round(last_lon, 2)
    start_cell = f"GRID_{grid_lat}_{grid_lon}"

    route_results = predict_routes(start_cell, steps=4)
    predicted_routes = [
        RouteStep(path=path, probability=round(prob, 6))
        for path, prob in route_results
    ]

    # 7. Priority Scoring
    anomaly_dicts = [ap.model_dump() for ap in anomaly_points]
    scored_areas = score_predicted_areas(
        predicted_areas=predicted_areas_raw,
        case_data=case_data,
        frequent_locations=freq_locations,
        route_predictions=route_results,
        anomaly_points=anomaly_dicts,
    )

    probable_areas = [ProbableArea(**area) for area in scored_areas]

    # 8. Explanation
    explanations_raw = explain_all_areas(scored_areas, case_data)
    explanations = []
    for exp in explanations_raw:
        explanations.append(AreaExplanation(
            area_name=exp["area_name"],
            priority_score=exp["priority_score"],
            priority_level=exp["priority_level"],
            ml_probability=exp["ml_probability"],
            summary=exp["summary"],
            supporting_factors=[
                ExplanationFactor(**f) for f in exp["supporting_factors"]
            ],
            top_model_features=[
                FeatureImportance(**f) for f in exp["top_model_features"]
            ],
        ))

    # 9. Final Summary
    top_area = scored_areas[0] if scored_areas else None
    summary_text = "Analysis complete. "
    if top_area:
        summary_text += (
            f"The highest-priority area is {top_area['area_name']} "
            f"with a {top_area['priority_level']} priority "
            f"(score: {top_area['priority_score']}). "
        )
    summary_text += (
        f"Analysis identified {len(freq_locations)} frequent locations, "
        f"{len(anomaly_points)} anomalous movement points, "
        f"{len(scored_areas)} probable areas, and "
        f"{len(predicted_routes)} predicted routes. "
        "All results are probabilistic investigation-support estimates "
        "and should not be treated as confirmed facts."
    )

    return AnalysisResponse(
        case_id=case_id,
        status="success",
        frequent_locations=[FrequentLocation(**loc) for loc in freq_locations],
        movement_patterns=movement_patterns,
        clusters=clusters,
        anomalies=anomaly_points,
        probable_areas=probable_areas,
        predicted_routes=predicted_routes,
        explanations=explanations,
        summary=summary_text,
    )
