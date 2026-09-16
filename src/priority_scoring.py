"""
Priority Scoring for CASEFILE.
Computes a weighted search-priority score for each predicted area using:
  ML Probability (30%), Historical Visit Frequency (20%), Route Similarity (15%),
  Distance Relevance (15%), Time Relevance (10%), Anomaly Evidence (10%).
Maps scores to priority bands: Low / Medium / High / Very High.
"""

import numpy as np


# Weights from README Section 8
WEIGHTS = {
    "ml_probability": 0.30,
    "visit_frequency": 0.20,
    "route_similarity": 0.15,
    "distance_relevance": 0.15,
    "time_relevance": 0.10,
    "anomaly_evidence": 0.10,
}

# Priority bands from README Section 8
PRIORITY_BANDS = [
    (0, 30, "Low"),
    (31, 60, "Medium"),
    (61, 80, "High"),
    (81, 100, "Very High"),
]


def get_priority_band(score):
    """Map a numeric score (0-100) to a priority band label."""
    score = max(0, min(100, score))
    for low, high, label in PRIORITY_BANDS:
        if low <= score <= high:
            return label
    return "Low"


def normalize_to_100(value, min_val=0.0, max_val=1.0):
    """Scale a value from [min_val, max_val] to [0, 100]."""
    if max_val <= min_val:
        return 50.0
    clamped = max(min_val, min(max_val, value))
    return ((clamped - min_val) / (max_val - min_val)) * 100.0


def compute_priority_score(
    ml_probability,
    visit_frequency_normalized,
    route_similarity_normalized,
    distance_relevance_normalized,
    time_relevance_normalized,
    anomaly_evidence_normalized,
):
    """
    Compute the weighted priority score for a predicted area.
    All inputs should be normalized to [0, 1] before calling.
    Returns (score_0_to_100, priority_band_label).
    """
    raw_score = (
        WEIGHTS["ml_probability"] * ml_probability
        + WEIGHTS["visit_frequency"] * visit_frequency_normalized
        + WEIGHTS["route_similarity"] * route_similarity_normalized
        + WEIGHTS["distance_relevance"] * distance_relevance_normalized
        + WEIGHTS["time_relevance"] * time_relevance_normalized
        + WEIGHTS["anomaly_evidence"] * anomaly_evidence_normalized
    )

    score_100 = round(raw_score * 100, 1)
    score_100 = max(0, min(100, score_100))
    band = get_priority_band(score_100)

    return score_100, band


def compute_distance_relevance(last_lat, last_lon, area_lat, area_lon):
    """
    Compute distance relevance: closer areas score higher.
    Uses inverse of Haversine distance, normalized.
    """
    R = 6371.0  # km
    phi1, phi2 = np.radians(last_lat), np.radians(area_lat)
    dphi = np.radians(area_lat - last_lat)
    dlam = np.radians(area_lon - last_lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    dist_km = R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Inverse distance scoring: closer = higher score, max 30km range
    max_dist = 30.0
    relevance = max(0.0, 1.0 - (dist_km / max_dist))
    return relevance


def compute_time_relevance(time_since_last_seen_hours):
    """
    Compute time relevance: more recent sightings score higher.
    Decays over 72 hours.
    """
    max_hours = 72.0
    if time_since_last_seen_hours <= 0:
        return 1.0
    relevance = max(0.0, 1.0 - (time_since_last_seen_hours / max_hours))
    return relevance


def score_predicted_areas(predicted_areas, case_data, frequent_locations,
                          route_predictions, anomaly_points):
    """
    Score all predicted areas for a case. Returns enriched list with scores.

    predicted_areas: list of (area_name, ml_probability) from location prediction
    case_data: dict with Last_Latitude, Last_Longitude, Time_Since_Last_Seen
    frequent_locations: list of dicts with grid_cell_id, visit_count, avg_lat, avg_lon
    route_predictions: list of (path, probability) from route prediction
    anomaly_points: list of dicts with latitude, longitude, anomaly_score
    """
    last_lat = float(case_data.get("Last_Latitude", 28.6139))
    last_lon = float(case_data.get("Last_Longitude", 77.2090))
    time_since = float(case_data.get("Time_Since_Last_Seen", 12.0))

    # Precompute visit frequency map (area_name -> normalized frequency)
    max_visits = max((loc["visit_count"] for loc in frequent_locations), default=1)
    visit_map = {}
    for loc in frequent_locations:
        visit_map[loc["grid_cell_id"]] = loc["visit_count"] / max_visits

    # Precompute route similarity: whether area appears in predicted routes
    route_cells = set()
    for path, _ in route_predictions:
        route_cells.update(path)

    # Area coordinate mapping from frequent locations
    area_coords = {}
    for loc in frequent_locations:
        area_coords[loc["grid_cell_id"]] = (loc["avg_lat"], loc["avg_lon"])

    # Known area center coordinates for the 6 Indian hubs
    area_centers = {
        "Connaught Place Hub": (28.6315, 77.2167),
        "Cyber City Gurgaon": (28.4950, 77.0890),
        "Noida Sector 18": (28.5708, 77.3261),
        "Hauz Khas Village": (28.5494, 77.2001),
        "Dwarka Sector 21": (28.5521, 77.0583),
        "Aerocity Hub": (28.5562, 77.1200),
    }

    time_rel = compute_time_relevance(time_since)

    scored_areas = []
    for area_name, ml_prob in predicted_areas:
        # Visit frequency for this area
        visit_freq = visit_map.get(area_name, 0.3)

        # Route similarity: is area on predicted route?
        route_sim = 0.7 if area_name in route_cells else 0.2

        # Distance relevance
        area_lat, area_lon = area_centers.get(area_name, (last_lat, last_lon))
        dist_rel = compute_distance_relevance(last_lat, last_lon, area_lat, area_lon)

        # Anomaly evidence near this area
        anomaly_near = 0.0
        for anom in anomaly_points:
            anom_lat = anom.get("latitude", 0)
            anom_lon = anom.get("longitude", 0)
            d = compute_distance_relevance(area_lat, area_lon, anom_lat, anom_lon)
            if d > 0.5:
                anomaly_near = max(anomaly_near, d)

        score, band = compute_priority_score(
            ml_probability=ml_prob,
            visit_frequency_normalized=visit_freq,
            route_similarity_normalized=route_sim,
            distance_relevance_normalized=dist_rel,
            time_relevance_normalized=time_rel,
            anomaly_evidence_normalized=anomaly_near,
        )

        scored_areas.append({
            "area_name": area_name,
            "ml_probability": round(float(ml_prob), 4),
            "visit_frequency": round(visit_freq, 4),
            "route_similarity": round(route_sim, 4),
            "distance_relevance": round(dist_rel, 4),
            "time_relevance": round(time_rel, 4),
            "anomaly_evidence": round(anomaly_near, 4),
            "priority_score": score,
            "priority_level": band,
            "area_latitude": area_lat,
            "area_longitude": area_lon,
        })

    scored_areas.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored_areas
