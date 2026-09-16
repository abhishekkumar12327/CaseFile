"""
Data Cleaning Module for CASEFILE
Cleans raw GeoLife GPS data: handles missing values, filters invalid coordinates,
removes duplicate records, parses timestamps, and creates temporal variables.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_GEOLIFE_FILE = os.path.join(DATA_DIR, 'raw', 'geolife', 'geolife_trajectories.csv')
PROCESSED_GPS_FILE = os.path.join(DATA_DIR, 'processed', 'cleaned_gps.csv')

def clean_gps_data(raw_filepath=RAW_GEOLIFE_FILE, output_filepath=PROCESSED_GPS_FILE):
    """
    Cleans raw GeoLife GPS records and performs temporal feature extraction.
    """
    print("Reading raw GPS trajectory data...")
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError(f"Raw data file not found at {raw_filepath}. Run data acquisition first.")
        
    df = pd.read_csv(raw_filepath)
    initial_count = len(df)
    
    # 1. Missing Value Handling
    df = df.dropna(subset=['user_id', 'trajectory_id', 'latitude', 'longitude', 'timestamp']).copy()
    
    # 2. Duplicate Removal
    df = df.drop_duplicates(subset=['user_id', 'trajectory_id', 'timestamp']).copy()
    
    # 3. Invalid Coordinate Removal (Target bounding box: Lat 39.5 - 40.5, Lon 116.0 - 117.0)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    valid_lat = (df['latitude'] >= 28.0) & (df['latitude'] <= 30.0)
    valid_lon = (df['longitude'] >= 76.0) & (df['longitude'] <= 78.0)
    
    df = df[valid_lat & valid_lon].copy()
    
    # 4. Timestamp Conversion & Temporal Features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).copy()
    
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday  # 0=Monday, 6=Sunday
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[-1, 5, 11, 17, 21, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Night'],
        ordered=False
    )
    
    # Sort chronologically per trajectory
    df = df.sort_values(by=['user_id', 'trajectory_id', 'timestamp']).reset_index(drop=True)
    
    cleaned_count = len(df)
    print(f"GPS Data Cleaning Complete. Rows before: {initial_count}, Rows after: {cleaned_count}")
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Cleaned GPS dataset saved to: {output_filepath}")
    return df

if __name__ == "__main__":
    clean_gps_data()
