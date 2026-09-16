"""
Data Acquisition Module for CASEFILE
Collects/Generates raw GeoLife GPS Trajectories and OpenStreetMap Geographic Features.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_GEOLIFE_DIR = os.path.join(DATA_DIR, 'raw', 'geolife')
RAW_OSM_DIR = os.path.join(DATA_DIR, 'raw', 'osm')

def ensure_directories():
    os.makedirs(RAW_GEOLIFE_DIR, exist_ok=True)
    os.makedirs(RAW_OSM_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'synthetic'), exist_ok=True)

def generate_geolife_raw_data(num_users=15, trajectories_per_user=10, points_per_trajectory=50):
    """
    Generates realistic GPS trajectory dataset formatted like GeoLife.
    Coordinates centered around India urban study area (Delhi-NCR: Lat 28.40 to 28.85, Lon 76.90 to 77.45).
    """
    ensure_directories()
    print("Generating raw GPS trajectory data (India / Delhi-NCR region)...")
    
    np.random.seed(42)
    start_date = datetime(2026, 4, 1, 8, 0, 0)
    
    records = []
    
    # Primary Indian urban hubs
    hubs = [
        {"name": "Connaught Place Hub", "lat": 28.6315, "lon": 77.2167},
        {"name": "Cyber City Gurgaon", "lat": 28.4950, "lon": 77.0890},
        {"name": "Noida Sector 18", "lat": 28.5708, "lon": 77.3261},
        {"name": "Hauz Khas Village", "lat": 28.5494, "lon": 77.2001},
        {"name": "Dwarka Sector 21", "lat": 28.5521, "lon": 77.0583},
        {"name": "Aerocity Hub", "lat": 28.5562, "lon": 77.1200},
        {"name": "Karol Bagh Center", "lat": 28.6514, "lon": 77.1907},
        {"name": "Saket District", "lat": 28.5244, "lon": 77.2183},
    ]

    for user_id in range(1, num_users + 1):
        user_str = f"User_{user_id:03d}"
        user_home = hubs[(user_id - 1) % len(hubs)]
        user_work = hubs[(user_id + 1) % len(hubs)]
        
        current_time = start_date + timedelta(days=user_id % 5, hours=user_id % 12)
        
        for traj_idx in range(1, trajectories_per_user + 1):
            traj_id = f"TRJ_{user_str}_{traj_idx:03d}"
            
            start_hub = user_home if traj_idx % 2 == 1 else user_work
            end_hub = user_work if traj_idx % 2 == 1 else hubs[np.random.randint(0, len(hubs))]
            
            curr_lat = start_hub["lat"] + np.random.normal(0, 0.003)
            curr_lon = start_hub["lon"] + np.random.normal(0, 0.003)
            
            lat_step = (end_hub["lat"] - curr_lat) / points_per_trajectory
            lon_step = (end_hub["lon"] - curr_lon) / points_per_trajectory
            
            for pt in range(points_per_trajectory):
                curr_lat += lat_step + np.random.normal(0, 0.0004)
                curr_lon += lon_step + np.random.normal(0, 0.0004)
                alt = float(np.round(np.random.uniform(200.0, 260.0), 1))
                
                sec_delta = np.random.randint(10, 60)
                current_time += timedelta(seconds=sec_delta)
                
                records.append({
                    "user_id": user_str,
                    "trajectory_id": traj_id,
                    "latitude": float(np.round(curr_lat, 6)),
                    "longitude": float(np.round(curr_lon, 6)),
                    "altitude_m": alt,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            current_time += timedelta(minutes=np.random.randint(30, 240))
            
    df = pd.DataFrame(records)
    
    outlier_idx = np.random.choice(df.index, 3, replace=False)
    df.loc[outlier_idx[0], "latitude"] = 999.0
    df.loc[outlier_idx[1], "longitude"] = -999.0
    
    output_path = os.path.join(RAW_GEOLIFE_DIR, 'geolife_trajectories.csv')
    df.to_csv(output_path, index=False)
    print(f"GeoLife Raw Data saved to: {output_path} ({len(df)} records)")
    return df

def generate_osm_raw_data(num_pois=200):
    """
    Generates realistic OpenStreetMap points of interest (POIs) across India study area.
    Categories include: Hospital, Metro Station, Market, School, Park, Police Station, Commercial Center.
    """
    ensure_directories()
    print("Generating raw OpenStreetMap POI & geographic data for India region...")
    
    np.random.seed(101)
    categories = [
        "Hospital", "Metro Station", "Railway Station", "Market", 
        "School", "Park", "Residential Area", "Road Intersection", "Police Station", "Commercial Center"
    ]
    
    districts = ["Central Delhi", "Gurgaon", "Noida", "South Delhi", "West Delhi", "North Delhi", "Faridabad"]
    
    records = []
    for i in range(1, num_pois + 1):
        cat = np.random.choice(categories)
        dist = np.random.choice(districts)
        
        # Lat range 28.40 to 28.85, Lon range 76.90 to 77.45
        lat = np.round(np.random.uniform(28.4000, 28.8500), 6)
        lon = np.round(np.random.uniform(76.9000, 77.4500), 6)
        
        records.append({
            "poi_id": f"OSM_IND_{i:04d}",
            "name": f"{dist} {cat} {i}",
            "category": cat,
            "district": dist,
            "latitude": lat,
            "longitude": lon,
            "amenity_type": cat.lower().replace(" ", "_")
        })
        
    df = pd.DataFrame(records)
    output_path = os.path.join(RAW_OSM_DIR, 'osm_features.csv')
    df.to_csv(output_path, index=False)
    print(f"OSM Raw Geographic Data saved to: {output_path} ({len(df)} records)")
    return df

if __name__ == "__main__":
    generate_geolife_raw_data()
    generate_osm_raw_data()
