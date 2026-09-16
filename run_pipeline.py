"""
CASEFILE Data Pipeline Execution Script
Executes all data acquisition, cleaning, feature engineering, and synthetic case generation steps.
"""

import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_acquisition import generate_geolife_raw_data, generate_osm_raw_data
from src.data_cleaning import clean_gps_data
from src.feature_engineering import create_movement_features, create_geographic_features
from src.synthetic_generator import generate_fictional_cases

def run_full_pipeline():
    print("==================================================")
    print("    STARTING CASEFILE SMART DATA PIPELINE        ")
    print("==================================================")
    
    # Step 1 & 2: Raw Data Acquisition
    print("\n[STEP 1 & 2] Downloading/Generating Raw Datasets...")
    df_raw_gps = generate_geolife_raw_data(num_users=15, trajectories_per_user=10, points_per_trajectory=50)
    df_raw_osm = generate_osm_raw_data(num_pois=200)
    
    # Step 3: GPS Data Cleaning
    print("\n[STEP 3] Cleaning Raw GPS Trajectory Data...")
    df_clean_gps = clean_gps_data()
    
    # Step 4 & 5: Feature Engineering & Geographical Context
    print("\n[STEP 4 & 5] Engineering Movement & Geographical Features...")
    df_mv = create_movement_features()
    df_geo = create_geographic_features()
    
    # Step 6: Synthetic Case Generation
    print("\n[STEP 6] Generating Fictional Investigation Cases...")
    df_cases = generate_fictional_cases(num_cases=100)
    
    print("\n==================================================")
    print("    DATA PIPELINE EXECUTED SUCCESSFULLY!          ")
    print("==================================================")
    print(f"Raw GPS Records:          {len(df_raw_gps)}")
    print(f"Cleaned GPS Records:      {len(df_clean_gps)}")
    print(f"Movement Features:        {len(df_mv)}")
    print(f"Geographic Features:      {len(df_geo)}")
    print(f"Synthetic Case Records:   {len(df_cases)}")
    print("==================================================")

if __name__ == "__main__":
    run_full_pipeline()
