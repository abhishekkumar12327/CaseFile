"""
Standalone training script for CASEFILE.
Trains all ML models (XGBoost location prediction, Isolation Forest anomaly detection,
scipy K-Means/Density clustering, Markov route matrix) and saves evaluation metrics.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.ml_pipeline import train_all_models

if __name__ == "__main__":
    metrics = train_all_models()
    print("\nTrained Metrics Summary:")
    print(f"Location Model Accuracy: {metrics['location_prediction']['accuracy']}")
    print(f"Location Model F1 (Macro): {metrics['location_prediction']['f1_macro']}")
    print(f"Location Top-1 / Top-3 / Top-5 Acc: {metrics['location_prediction']['top_1_accuracy']} / {metrics['location_prediction']['top_3_accuracy']} / {metrics['location_prediction']['top_5_accuracy']}")
    print(f"Anomaly Detection FPR: {metrics['anomaly_detection']['false_positive_rate']}, FNR: {metrics['anomaly_detection']['false_negative_rate']}")
