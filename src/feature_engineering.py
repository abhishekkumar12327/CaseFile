"""
Feature Engineering Module for CASEFILE
Engineers movement features (distance, speed, visit frequency, stay-time, grid locations)
and geographical features (POI proximity, spatial density, district mapping).
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CLEANED_GPS_FILE = os.path.join(DATA_DIR, 'processed', 'cleaned_gps.csv')
OSM_FILE = os.path.join(DATA_DIR, 'raw', 'osm', 'osm_features.csv')

MOVEMENT_FEATURES_FILE = os.path.join(DATA_DIR, 'processed', 'movement_features.csv')
GEOGRAPHIC_FEATURES_FILE = os.path.join(DATA_DIR, 'processed', 'geographic_features.csv')

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes Haversine distance between point pairs in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def create_movement_features(gps_file=CLEANED_GPS_FILE, output_file=MOVEMENT_FEATURES_FILE):
    """
    Calculates consecutive point distances, speeds, trajectory aggregations, stay times, and visit counts.
    """
    print("Creating movement features...")
    df = pd.read_csv(gps_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort chronologically per user and trajectory
    df = df.sort_values(by=['user_id', 'trajectory_id', 'timestamp']).reset_index(drop=True)
    
    # Lead/Lag fields for consecutive calculations within same trajectory
    df['prev_lat'] = df.groupby(['user_id', 'trajectory_id'])['latitude'].shift(1)
    df['prev_lon'] = df.groupby(['user_id', 'trajectory_id'])['longitude'].shift(1)
    df['prev_time'] = df.groupby(['user_id', 'trajectory_id'])['timestamp'].shift(1)
    
    # Distance in meters
    df['step_distance_m'] = haversine_distance(
        df['latitude'].values, df['longitude'].values,
        df['prev_lat'].values, df['prev_lon'].values
    )
    df['step_distance_m'] = df['step_distance_m'].fillna(0.0)
    
    # Time diff in seconds
    df['step_time_sec'] = (df['timestamp'] - df['prev_time']).dt.total_seconds().fillna(0.0)
    
    # Instantaneous speed (km/h)
    # Avoid div by zero
    speed_ms = np.where(df['step_time_sec'] > 0, df['step_distance_m'] / df['step_time_sec'], 0.0)
    df['instant_speed_kmh'] = np.round(speed_ms * 3.6, 2)
    # Cap unrealistic speed spikes (> 180 km/h)
    df['instant_speed_kmh'] = np.clip(df['instant_speed_kmh'], 0, 180)
    
    # Spatial Grid ID (0.01 deg resolution ~ 1km grid)
    df['grid_lat'] = np.round(df['latitude'], 2)
    df['grid_lon'] = np.round(df['longitude'], 2)
    df['grid_cell_id'] = "GRID_" + df['grid_lat'].astype(str) + "_" + df['grid_lon'].astype(str)
    
    # Trajectory-level summary statistics
    traj_summary = df.groupby(['user_id', 'trajectory_id']).agg(
        total_distance_m=('step_distance_m', 'sum'),
        avg_speed_kmh=('instant_speed_kmh', 'mean'),
        max_speed_kmh=('instant_speed_kmh', 'max'),
        duration_minutes=('step_time_sec', lambda x: x.sum() / 60.0),
        locations_visited_count=('grid_cell_id', 'nunique'),
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max')
    ).reset_index()
    
    # Merge summary metrics back to point level
    df = df.merge(traj_summary, on=['user_id', 'trajectory_id'], suffixes=('', '_traj'))
    
    # User-level grid visit frequency
    grid_visits = df.groupby(['user_id', 'grid_cell_id']).size().reset_index(name='user_grid_visit_count')
    df = df.merge(grid_visits, on=['user_id', 'grid_cell_id'], how='left')
    
    # Cleanup temporary columns
    df = df.drop(columns=['prev_lat', 'prev_lon', 'prev_time'])
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Movement Features saved to: {output_file} ({len(df)} records)")
    return df

def create_geographic_features(gps_file=CLEANED_GPS_FILE, osm_file=OSM_FILE, output_file=GEOGRAPHIC_FEATURES_FILE):
    """
    Enriches GPS coordinates with nearest OSM POI proximity, POI count density, and land-use categories.
    """
    print("Creating geographic features...")
    df_gps = pd.read_csv(gps_file)
    
    if not os.path.exists(osm_file):
        print("OSM file not found. Generating default OSM raw dataset...")
        from src.data_acquisition import generate_osm_raw_data
        df_osm = generate_osm_raw_data()
    else:
        df_osm = pd.read_csv(osm_file)
        
    osm_lats = df_osm['latitude'].values
    osm_lons = df_osm['longitude'].values
    osm_cats = df_osm['category'].values
    osm_names = df_osm['name'].values
    
    nearest_poi_names = []
    nearest_poi_cats = []
    nearest_poi_dist_m = []
    pois_within_500m_count = []
    hospital_within_1km_flag = []
    transit_within_500m_flag = []
    
    for idx, row in df_gps.iterrows():
        lat, lon = row['latitude'], row['longitude']
        
        # Calculate distances to all OSM POIs
        dists = haversine_distance(lat, lon, osm_lats, osm_lons)
        
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        nearest_poi_names.append(osm_names[min_idx])
        nearest_poi_cats.append(osm_cats[min_idx])
        nearest_poi_dist_m.append(np.round(min_dist, 1))
        
        # POIs within 500m
        within_500m = dists <= 500.0
        pois_within_500m_count.append(int(np.sum(within_500m)))
        
        # Hospital flag within 1km
        hosp_mask = (osm_cats == "Hospital") & (dists <= 1000.0)
        hospital_within_1km_flag.append(1 if np.any(hosp_mask) else 0)
        
        # Transit flag within 500m
        transit_mask = (np.isin(osm_cats, ["Bus Station", "Subway Station"])) & (dists <= 500.0)
        transit_within_500m_flag.append(1 if np.any(transit_mask) else 0)
        
    geo_df = df_gps.copy()
    geo_df['nearest_poi_name'] = nearest_poi_names
    geo_df['nearest_poi_category'] = nearest_poi_cats
    geo_df['nearest_poi_dist_m'] = nearest_poi_dist_m
    geo_df['poi_density_500m'] = pois_within_500m_count
    geo_df['has_hospital_1km'] = hospital_within_1km_flag
    geo_df['has_transit_500m'] = transit_within_500m_flag
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    geo_df.to_csv(output_file, index=False)
    print(f"Geographic Features saved to: {output_file} ({len(geo_df)} records)")
    return geo_df

if __name__ == "__main__":
    create_movement_features()
    create_geographic_features()
