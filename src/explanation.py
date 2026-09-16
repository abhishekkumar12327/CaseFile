"""
Model Explanation for CASEFILE.
Generates human-readable explanations for why each area received its
prediction and priority score, using feature importance from the trained model.
"""

import os
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def get_feature_importance():
    """Load feature importance from the trained location prediction model."""
    importance_path = os.path.join(MODELS_DIR, "feature_importance.joblib")
    if os.path.exists(importance_path):
        return joblib.load(importance_path)
    return {}


def explain_area_prediction(scored_area, case_data):
    """
    Generate a human-readable explanation for why an area received its
    priority score and prediction probability.

    scored_area: dict from priority_scoring.score_predicted_areas
    case_data: original case input dict
    """
    area_name = scored_area["area_name"]
    ml_prob = scored_area["ml_probability"]
    score = scored_area["priority_score"]
    band = scored_area["priority_level"]

    factors = []

    # ML probability factor
    if ml_prob > 0.3:
        factors.append({
            "factor": "High model prediction confidence",
            "detail": f"The ML model assigned a {ml_prob:.1%} probability to this area.",
            "weight": "30%",
            "contribution": "strong"
        })
    elif ml_prob > 0.15:
        factors.append({
            "factor": "Moderate model prediction",
            "detail": f"The ML model assigned a {ml_prob:.1%} probability to this area.",
            "weight": "30%",
            "contribution": "moderate"
        })
    else:
        factors.append({
            "factor": "Low model prediction",
            "detail": f"The ML model assigned a {ml_prob:.1%} probability to this area.",
            "weight": "30%",
            "contribution": "weak"
        })

    # Visit frequency factor
    visit_freq = scored_area.get("visit_frequency", 0)
    if visit_freq > 0.6:
        factors.append({
            "factor": "High historical visit frequency",
            "detail": "This area was frequently visited in historical movement data.",
            "weight": "20%",
            "contribution": "strong"
        })
    elif visit_freq > 0.3:
        factors.append({
            "factor": "Moderate historical visit frequency",
            "detail": "This area was visited occasionally in historical movement data.",
            "weight": "20%",
            "contribution": "moderate"
        })

    # Route similarity factor
    route_sim = scored_area.get("route_similarity", 0)
    if route_sim > 0.5:
        factors.append({
            "factor": "Matches predicted movement route",
            "detail": "This area lies along a probable movement route from the last known location.",
            "weight": "15%",
            "contribution": "strong"
        })

    # Distance relevance factor
    dist_rel = scored_area.get("distance_relevance", 0)
    if dist_rel > 0.6:
        factors.append({
            "factor": "Close to last known location",
            "detail": "This area is geographically close to where the person was last observed.",
            "weight": "15%",
            "contribution": "strong"
        })
    elif dist_rel > 0.3:
        factors.append({
            "factor": "Moderate distance from last location",
            "detail": "This area is at a moderate distance from the last known location.",
            "weight": "15%",
            "contribution": "moderate"
        })

    # Time relevance factor
    time_rel = scored_area.get("time_relevance", 0)
    time_since = case_data.get("Time_Since_Last_Seen", 0)
    if time_rel > 0.7:
        factors.append({
            "factor": "Recent last observation",
            "detail": f"The person was last seen {time_since:.1f} hours ago, making nearby predictions more relevant.",
            "weight": "10%",
            "contribution": "strong"
        })

    # Anomaly evidence factor
    anomaly_ev = scored_area.get("anomaly_evidence", 0)
    if anomaly_ev > 0.3:
        factors.append({
            "factor": "Anomalous movement detected nearby",
            "detail": "Unusual movement patterns were detected near this area.",
            "weight": "10%",
            "contribution": "moderate"
        })

    # Get model feature importance
    feature_importance = get_feature_importance()
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

    feature_names_readable = {
        "Last_Latitude": "Last known latitude",
        "Last_Longitude": "Last known longitude",
        "Age_Group_enc": "Age group",
        "Day_enc": "Day of the week",
        "Weather_enc": "Weather conditions",
        "Average_Distance": "Average travel distance",
        "Average_Speed": "Average movement speed",
        "Usual_Area_enc": "Usual area",
        "Previous_Area_enc": "Previous area visited",
        "Time_Since_Last_Seen": "Time since last seen",
    }

    top_feature_explanations = []
    for feat_name, importance in top_features:
        readable = feature_names_readable.get(feat_name, feat_name)
        top_feature_explanations.append({
            "feature": readable,
            "importance": round(importance, 4),
        })

    return {
        "area_name": area_name,
        "priority_score": score,
        "priority_level": band,
        "ml_probability": ml_prob,
        "summary": (
            f"{area_name} received a {band} priority (score: {score}) "
            f"based on a combination of model prediction ({ml_prob:.1%}), "
            f"historical visit patterns, route analysis, geographic proximity, "
            f"and temporal factors. All results are probabilistic estimates."
        ),
        "supporting_factors": factors,
        "top_model_features": top_feature_explanations,
    }


def explain_all_areas(scored_areas, case_data):
    """Generate explanations for all scored areas."""
    return [explain_area_prediction(area, case_data) for area in scored_areas]
