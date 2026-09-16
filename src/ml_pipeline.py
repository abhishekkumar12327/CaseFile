"""
ML Pipeline for CASEFILE.
Provides clustering, anomaly detection, location prediction, and route prediction.
All models are trained on processed movement data and saved with Joblib.

Tech Stack Specifications & Justifications:
- Location Prediction: XGBoost (xgboost.XGBClassifier) per TECH STACK.md.
- Anomaly Detection: Isolation Forest (sklearn.ensemble.IsolationForest).
- Route Prediction: Markov Transition Matrix.
- Clustering: Uses scipy.cluster.vq for K-Means and scipy.spatial distance matrix
  density clustering. (Explicit Tech-Stack Deviation Flag: Necessary because
  sklearn.cluster._k_means_common.pyd is blocked on this host environment
  by Windows OS Application Control Policy).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2, whiten
from scipy.spatial.distance import cdist
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Movement Clustering (scipy K-Means + density-based grouping)
# ---------------------------------------------------------------------------

def _density_cluster(X, eps=0.5, min_samples=10):
    """Simple density-based clustering (DBSCAN-like) using distance matrix."""
    n = len(X)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    visited = np.zeros(n, dtype=bool)
    dists = cdist(X, X)

    for i in range(n):
        if visited[i]:
            continue
        neighbors = np.where(dists[i] <= eps)[0]
        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        visited[i] = True
        seed_set = list(neighbors)

        while seed_set:
            j = seed_set.pop(0)
            if not visited[j]:
                visited[j] = True
                j_neighbors = np.where(dists[j] <= eps)[0]
                if len(j_neighbors) >= min_samples:
                    seed_set.extend(j_neighbors.tolist())
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels


def train_clustering_models(movement_features_path=None):
    """Train K-Means and density clustering on spatial + movement features."""
    if movement_features_path is None:
        movement_features_path = os.path.join(DATA_DIR, "processed", "movement_features.csv")

    df = pd.read_csv(movement_features_path)
    cluster_features = ["latitude", "longitude", "instant_speed_kmh", "step_distance_m"]
    X = df[cluster_features].dropna().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means via scipy
    centroids, kmeans_labels = kmeans2(X_scaled, k=6, minit="points", seed=42)

    # Density clustering on a subsample
    subsample_size = min(1000, len(X_scaled))
    subsample_idx = np.random.RandomState(42).choice(len(X_scaled), subsample_size, replace=False)
    X_sub = X_scaled[subsample_idx]
    dbscan_labels_sub = _density_cluster(X_sub, eps=0.8, min_samples=5)

    # Assign density labels to full dataset via nearest centroid from subsample clusters
    unique_clusters = [c for c in np.unique(dbscan_labels_sub) if c != -1]
    dbscan_labels = np.full(len(X_scaled), -1, dtype=int)
    if unique_clusters:
        cluster_centers = np.array([X_sub[dbscan_labels_sub == c].mean(axis=0) for c in unique_clusters])
        dists_to_centers = cdist(X_scaled, cluster_centers)
        nearest = dists_to_centers.argmin(axis=1)
        min_dists = dists_to_centers.min(axis=1)
        dbscan_labels = np.where(min_dists < 2.0, nearest, -1)

    ensure_models_dir()
    joblib.dump({"centroids": centroids, "k": 6}, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "cluster_scaler.joblib"))

    n_kmeans = len(set(kmeans_labels))
    n_density = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"Clustering trained — K-Means clusters: {n_kmeans}, Density clusters: {n_density}")

    return centroids, scaler


def predict_clusters(points_df):
    """Assign cluster labels to GPS points using saved K-Means centroids."""
    model_data = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "cluster_scaler.joblib"))
    centroids = model_data["centroids"]

    features = ["latitude", "longitude", "instant_speed_kmh", "step_distance_m"]
    X = points_df[features].fillna(0).values
    X_scaled = scaler.transform(X)

    dists = cdist(X_scaled, centroids)
    labels = dists.argmin(axis=1)
    return labels


# ---------------------------------------------------------------------------
# 2. Anomaly Detection (Isolation Forest)
# ---------------------------------------------------------------------------

def train_anomaly_model(movement_features_path=None):
    """Train Isolation Forest on movement features to flag unusual patterns."""
    if movement_features_path is None:
        movement_features_path = os.path.join(DATA_DIR, "processed", "movement_features.csv")

    df = pd.read_csv(movement_features_path)
    anomaly_features = ["latitude", "longitude", "instant_speed_kmh",
                        "step_distance_m", "step_time_sec"]
    X = df[anomaly_features].dropna().values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso_forest.fit(X_scaled)

    labels = iso_forest.predict(X_scaled)
    anomaly_count = int((labels == -1).sum())

    # Evaluation metrics for Anomaly Detection
    # Synthetic normal points vs artificial outliers for evaluation
    rng = np.random.RandomState(42)
    n_eval = 200
    normal_eval = X_scaled[rng.choice(len(X_scaled), n_eval // 2, replace=False)]
    outlier_eval = rng.uniform(low=-5.0, high=5.0, size=(n_eval // 2, X_scaled.shape[1]))

    eval_X = np.vstack([normal_eval, outlier_eval])
    eval_y_true = np.array([1] * (n_eval // 2) + [-1] * (n_eval // 2))
    eval_y_pred = iso_forest.predict(eval_X)

    # False Positive Rate: Normal points classified as Anomaly (-1)
    fp = np.sum((eval_y_true == 1) & (eval_y_pred == -1))
    tn = np.sum((eval_y_true == 1) & (eval_y_pred == 1))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # False Negative Rate: Outliers (-1) classified as Normal (1)
    fn = np.sum((eval_y_true == -1) & (eval_y_pred == 1))
    tp = np.sum((eval_y_true == -1) & (eval_y_pred == -1))
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    anomaly_metrics = {
        "contamination_rate": 0.05,
        "anomalies_flagged_count": anomaly_count,
        "total_points_trained": len(X_scaled),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
    }

    ensure_models_dir()
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "anomaly_scaler.joblib"))

    print(f"Anomaly detection trained — {anomaly_count} anomalies "
          f"({anomaly_count / len(labels) * 100:.1f}%), FPR: {fpr:.4f}, FNR: {fnr:.4f}")

    return iso_forest, scaler, anomaly_metrics


def detect_anomalies(points_df):
    """Score new points for anomalies. Returns labels (-1=anomaly, 1=normal) and scores."""
    iso_forest = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "anomaly_scaler.joblib"))

    features = ["latitude", "longitude", "instant_speed_kmh",
                "step_distance_m", "step_time_sec"]
    X = points_df[features].fillna(0).values
    X_scaled = scaler.transform(X)

    labels = iso_forest.predict(X_scaled)
    scores = iso_forest.decision_function(X_scaled)
    return labels, scores


# ---------------------------------------------------------------------------
# 3. Location Prediction (XGBoost Classifier)
# ---------------------------------------------------------------------------

def compute_top_k_accuracy(y_true, y_prob, k):
    """Compute Top-k accuracy for multi-class classification probabilities."""
    n_samples = len(y_true)
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    hits = 0
    for i in range(n_samples):
        if y_true[i] in top_k_preds[i]:
            hits += 1
    return float(hits / n_samples)


def train_location_model(cases_path=None):
    """Train XGBoost classifier to predict Target_Area from case features."""
    if cases_path is None:
        cases_path = os.path.join(DATA_DIR, "synthetic", "fictional_cases.csv")

    df = pd.read_csv(cases_path)

    le_age = LabelEncoder()
    le_day = LabelEncoder()
    le_weather = LabelEncoder()
    le_usual = LabelEncoder()
    le_prev = LabelEncoder()
    le_target = LabelEncoder()

    df["Age_Group_enc"] = le_age.fit_transform(df["Age_Group"])
    df["Day_enc"] = le_day.fit_transform(df["Day"])
    df["Weather_enc"] = le_weather.fit_transform(df["Weather"])
    df["Usual_Area_enc"] = le_usual.fit_transform(df["Usual_Area"])
    df["Previous_Area_enc"] = le_prev.fit_transform(df["Previous_Area"])
    df["Target_Area_enc"] = le_target.fit_transform(df["Target_Area"])

    feature_cols = ["Last_Latitude", "Last_Longitude", "Age_Group_enc",
                    "Day_enc", "Weather_enc", "Average_Distance",
                    "Average_Speed", "Usual_Area_enc", "Previous_Area_enc",
                    "Time_Since_Last_Seen"]
    X = df[feature_cols].values
    y = df["Target_Area_enc"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use XGBoost Classifier per TECH STACK.md
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    # Calculate metrics required by README.md & evaluation requirements
    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred).tolist()

    n_classes = len(np.unique(y))
    top1_acc = compute_top_k_accuracy(y_test, y_prob, k=min(1, n_classes))
    top3_acc = compute_top_k_accuracy(y_test, y_prob, k=min(3, n_classes))
    top5_acc = compute_top_k_accuracy(y_test, y_prob, k=min(5, n_classes))

    location_metrics = {
        "model_type": "XGBoost Classifier",
        "accuracy": round(accuracy, 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "top_1_accuracy": round(top1_acc, 4),
        "top_3_accuracy": round(top3_acc, 4),
        "top_5_accuracy": round(top5_acc, 4),
        "classes": le_target.classes_.tolist(),
        "confusion_matrix": cm,
    }

    print(f"Location Prediction (XGBoost) — Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Top-3 Acc: {top3_acc:.3f}")

    feature_importance = dict(zip(feature_cols, model.feature_importances_.tolist()))

    ensure_models_dir()
    joblib.dump(model, os.path.join(MODELS_DIR, "location_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "location_scaler.joblib"))
    joblib.dump(le_target, os.path.join(MODELS_DIR, "target_encoder.joblib"))
    joblib.dump(feature_importance, os.path.join(MODELS_DIR, "feature_importance.joblib"))

    encoders = {
        "age": le_age, "day": le_day, "weather": le_weather,
        "usual": le_usual, "previous": le_prev, "target": le_target
    }
    joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.joblib"))

    return model, scaler, encoders, feature_importance, location_metrics


def predict_location(case_data: dict):
    """
    Predict probable areas for a case. Returns list of (area_name, probability) tuples.
    """
    model = joblib.load(os.path.join(MODELS_DIR, "location_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "location_scaler.joblib"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.joblib"))
    le_target = joblib.load(os.path.join(MODELS_DIR, "target_encoder.joblib"))

    def safe_encode(encoder, value):
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        return 0

    features = np.array([[
        float(case_data.get("Last_Latitude", 28.6139)),
        float(case_data.get("Last_Longitude", 77.2090)),
        safe_encode(encoders["age"], case_data.get("Age_Group", "Adult (36-60)")),
        safe_encode(encoders["day"], case_data.get("Day", "Monday")),
        safe_encode(encoders["weather"], case_data.get("Weather", "Clear")),
        float(case_data.get("Average_Distance", 5.0)),
        float(case_data.get("Average_Speed", 15.0)),
        safe_encode(encoders["usual"], case_data.get("Usual_Area", "Connaught Place Hub")),
        safe_encode(encoders["previous"], case_data.get("Previous_Area", "Cyber City Gurgaon")),
        float(case_data.get("Time_Since_Last_Seen", 12.0)),
    ]])

    features_scaled = scaler.transform(features)
    probabilities = model.predict_proba(features_scaled)[0]
    classes = le_target.inverse_transform(model.classes_)

    results = sorted(zip(classes, probabilities.tolist()), key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 4. Route Prediction (Markov Transition Matrix)
# ---------------------------------------------------------------------------

def train_route_model(movement_features_path=None):
    """Build a Markov transition matrix from trajectory grid-cell transitions."""
    if movement_features_path is None:
        movement_features_path = os.path.join(DATA_DIR, "processed", "movement_features.csv")

    df = pd.read_csv(movement_features_path)
    df = df.sort_values(by=["user_id", "trajectory_id", "start_time"])

    df["next_grid"] = df.groupby(["user_id", "trajectory_id"])["grid_cell_id"].shift(-1)
    transitions = df.dropna(subset=["next_grid"])[["grid_cell_id", "next_grid"]].copy()

    transition_counts = transitions.groupby(["grid_cell_id", "next_grid"]).size().reset_index(name="count")
    totals = transition_counts.groupby("grid_cell_id")["count"].transform("sum")
    transition_counts["probability"] = transition_counts["count"] / totals

    transition_matrix = {}
    for _, row in transition_counts.iterrows():
        src = row["grid_cell_id"]
        dst = row["next_grid"]
        prob = row["probability"]
        if src not in transition_matrix:
            transition_matrix[src] = {}
        transition_matrix[src][dst] = prob

    ensure_models_dir()
    joblib.dump(transition_matrix, os.path.join(MODELS_DIR, "transition_matrix.joblib"))

    print(f"Route model trained — {len(transition_matrix)} source cells, "
          f"{len(transition_counts)} transitions")

    route_metrics = {
        "source_grid_cells": len(transition_matrix),
        "total_transitions_mapped": len(transition_counts),
    }

    return transition_matrix, route_metrics


def predict_routes(start_grid_cell, steps=5):
    """Predict probable routes from a start grid cell using beam search on the Markov matrix."""
    transition_matrix = joblib.load(os.path.join(MODELS_DIR, "transition_matrix.joblib"))

    if start_grid_cell not in transition_matrix:
        available = list(transition_matrix.keys())
        if not available:
            return []
        start_grid_cell = available[0]

    beam_width = 3
    routes = [([start_grid_cell], 1.0)]

    for _ in range(steps):
        candidates = []
        for path, prob in routes:
            current = path[-1]
            if current in transition_matrix:
                next_cells = sorted(transition_matrix[current].items(),
                                    key=lambda x: x[1], reverse=True)[:beam_width]
                for next_cell, trans_prob in next_cells:
                    candidates.append((path + [next_cell], prob * trans_prob))
            else:
                candidates.append((path, prob))

        candidates.sort(key=lambda x: x[1], reverse=True)
        routes = candidates[:beam_width]

    return routes


# ---------------------------------------------------------------------------
# 5. Get Frequent Locations
# ---------------------------------------------------------------------------

def get_frequent_locations(user_id=None, top_n=10):
    """Get most frequently visited grid cells from movement data."""
    movement_path = os.path.join(DATA_DIR, "processed", "movement_features.csv")
    df = pd.read_csv(movement_path)

    if user_id is not None:
        df = df[df["user_id"] == user_id]

    if df.empty:
        return []

    freq = df.groupby("grid_cell_id").agg(
        visit_count=("grid_cell_id", "size"),
        avg_lat=("latitude", "mean"),
        avg_lon=("longitude", "mean"),
        avg_speed=("instant_speed_kmh", "mean"),
        total_time_min=("step_time_sec", lambda x: x.sum() / 60.0)
    ).reset_index()

    freq = freq.sort_values("visit_count", ascending=False).head(top_n)
    return freq.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Train All Models & Export Evaluation Metrics
# ---------------------------------------------------------------------------

def train_all_models():
    """Train and save all ML models, returning comprehensive evaluation metrics."""
    print("=" * 50)
    print("    TRAINING ALL ML MODELS (XGBoost & Evaluation Metrics)")
    print("=" * 50)

    print("\n[1/4] Training clustering models...")
    centroids, _ = train_clustering_models()

    print("\n[2/4] Training anomaly detection model...")
    _, _, anomaly_metrics = train_anomaly_model()

    print("\n[3/4] Training location prediction model (XGBoost)...")
    _, _, _, _, location_metrics = train_location_model()

    print("\n[4/4] Training route prediction model...")
    _, route_metrics = train_route_model()

    metrics = {
        "location_prediction": location_metrics,
        "anomaly_detection": anomaly_metrics,
        "route_prediction": route_metrics,
        "clustering_notes": (
            "Uses scipy K-Means & spatial density clustering due to Windows OS "
            "Application Control policy blocking sklearn.cluster._k_means_common.pyd"
        ),
    }

    ensure_models_dir()
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 50)
    print(f"    ALL MODELS TRAINED SUCCESSFULLY — Metrics saved to {metrics_path}")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    train_all_models()
