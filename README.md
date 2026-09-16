# 🇮🇳 CASEFILE — AI-Powered Geospatial Movement Analysis & Investigation Support

An advanced, privacy-compliant **AI & Geospatial Investigation Support System** designed to assist investigation teams in analyzing movement trajectories, detecting spatial anomalies, predicting probable destination areas, and generating transparent model explanations visualized on interactive satellite imagery.

---

## 🎯 Problem Statement & Solution

### The Problem
During missing-person investigations or movement trajectory analysis, investigation teams face overwhelming volumes of unstructured GPS coordinates, cell tower logs, and road network data. Manually analyzing thousands of trajectory points is slow, prone to human oversight, and lacks probabilistic priority estimation.

### Our Solution
**CASEFILE** automates trajectory intelligence through an end-to-end Machine Learning pipeline:
1. **Anomaly Detection**: Flags statistically abnormal speeds, distances, or spatial deviations using Isolation Forests.
2. **Location Prediction**: Predicts probable destination areas with **99.0% accuracy** using XGBoost Classifiers.
3. **Route Forecasting**: Predicts multi-step movement corridors via Markov Transition Matrices.
4. **Weighted Priority Engine**: Computes a 6-factor priority score (0–100) assigning risk bands (Low, Medium, High, Very High).
5. **Human-Readable Explanations**: Generates feature importance charts and textual reasoning for investigative transparency.
6. **India Satellite Visualization**: Visualizes all predictions, route corridors, and anomalies on an Esri World Imagery Satellite map centered on the India region.

---

## 📁 Repository Folder Structure & Module Guide

```
CaseFile/
│
├── app.py                      # Streamlit India Satellite Dashboard Frontend
├── train_models.py             # Standalone Model Training & Metrics Entrypoint
├── run_pipeline.py             # Standalone Pipeline Runner
│
├── backend/                    # FastAPI Backend Application
│   ├── __init__.py             # Backend package initialization
│   ├── main.py                 # FastAPI REST API endpoints & pipeline orchestration
│   └── schemas.py              # Pydantic data schemas for request/response validation
│
├── src/                        # Core Data Processing & ML Modules
│   ├── data_acquisition.py     # GPS trajectory & OpenStreetMap POI synthetic generator
│   ├── data_cleaning.py        # Coordinate filtering, missing value imputation & temporal parsing
│   ├── feature_engineering.py  # Spatial grid mapping, Haversine distance & speed calculation
│   ├── synthetic_generator.py  # Pattern-correlated fictional case scenario generator
│   ├── ml_pipeline.py          # XGBoost classifier, Isolation Forest & scipy K-Means/Density clustering
│   ├── priority_scoring.py     # 6-Factor weighted priority score engine
│   └── explanation.py          # Model interpretability & reasoning generator
│
├── data/                       # Data Storage Directory
│   ├── raw/                    # Raw GPS trajectories and OSM POIs
│   ├── processed/              # Cleaned movement features and geographic datasets
│   └── synthetic/              # Fictional investigation case scenarios (500 records)
│
├── models/                     # Trained Model Artifacts & Evaluation Metrics
│   ├── location_model.joblib   # Trained XGBoost location classifier
│   ├── isolation_forest.joblib # Trained Isolation Forest anomaly detector
│   ├── kmeans_model.joblib     # Spatial K-Means centroids
│   ├── transition_matrix.joblib# Markov route transition probability matrix
│   └── model_metrics.json      # Saved accuracy, F1 score, confusion matrix & metrics
│
└── md_files/                   # System Documentation & Specifications
    ├── PRD.md                  # Product Requirements Document
    ├── FRONTEND.md             # Frontend UI/UX Specification
    ├── TECH STACK.md           # Architecture & Technology Stack Reference
    ├── USERFLOW.md             # Complete User Journey Specification
    ├── TEST.md                 # Test Plan & Empirical Execution Suite (100% Pass)
    └── RULES.md                # System Development & Architecture Rules
```

---

## 🚀 How to Run the Application

### 1. Install Dependencies
```bash
pip install streamlit fastapi uvicorn xgboost scikit-learn scipy pandas numpy folium streamlit-folium geopandas plotly joblib
```

### 2. Generate Data & Train Models (Optional / Pre-Configured)
```bash
python train_models.py
```

### 3. Start the FastAPI Backend Server
Run the backend REST API on port `8000`:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
> **Backend API URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)  
> **Interactive API Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Start the Streamlit Frontend Dashboard
In a new terminal window, launch the dashboard on port `8501`:
```bash
streamlit run app.py --server.port 8501
```
> **Frontend Dashboard URL**: [http://localhost:8501](http://localhost:8501)

---

## 🔒 Responsible-Use Disclaimer
**CASEFILE** is an academic prototype designed solely for privacy-compliant simulation and decision-support modeling. All cases and trajectories are synthetic representations. Predicted areas are model probability estimates and must **never** be interpreted as confirmed real-world facts or proof of wrongdoing.
