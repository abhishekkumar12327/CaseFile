"""
Synthetic Case Generator Module for CASEFILE
Generates privacy-compliant fictional missing-person investigation records
matching Section 3 Step 6 of CASEFILE Smart Data Acquisition Plan.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
SYNTHETIC_FILE = os.path.join(DATA_DIR, 'synthetic', 'fictional_cases.csv')

def generate_fictional_cases(num_cases=500, output_file=SYNTHETIC_FILE):
    """
    Generates controlled synthetic investigation scenarios using realistic trajectory parameters
    and correlated movement target areas (e.g., usual area affinity, spatial proximity, age patterns).
    No real missing-person data or PII is used.
    """
    print(f"Generating {num_cases} synthetic fictional investigation case data...")
    np.random.seed(2026)
    
    age_groups = ["Child (5-12)", "Teen (13-17)", "Young Adult (18-35)", "Adult (36-60)", "Senior (60+)"]
    weather_options = ["Clear", "Rainy", "Foggy", "Overcast", "Windy"]
    
    area_coords = {
        "Connaught Place Hub": (28.6315, 77.2167),
        "Cyber City Gurgaon": (28.4950, 77.0890),
        "Noida Sector 18": (28.5708, 77.3261),
        "Hauz Khas Village": (28.5494, 77.2001),
        "Dwarka Sector 21": (28.5521, 77.0583),
        "Aerocity Hub": (28.5562, 77.1200),
    }
    areas = list(area_coords.keys())
    
    records = []
    base_date = datetime(2026, 9, 1, 10, 0, 0)
    
    for i in range(1, num_cases + 1):
        case_id = f"CASE_IND_{i:04d}"
        person_id = f"PERSON_IND_{i:04d}"
        age = np.random.choice(age_groups, p=[0.15, 0.15, 0.35, 0.25, 0.10])
        
        # Select usual area and previous area
        usual = np.random.choice(areas)
        prev = np.random.choice([a for a in areas if a != usual] + [usual])
        
        # Correlate Target_Area with usual area (45%), previous area (25%), or spatial proximity (30%)
        prob_choice = np.random.rand()
        if prob_choice < 0.45:
            target = usual
        elif prob_choice < 0.70:
            target = prev
        else:
            target = np.random.choice(areas)
            
        # Coordinates centered around target area with slight spatial spread
        target_lat, target_lon = area_coords[target]
        lat = np.round(target_lat + np.random.normal(0, 0.015), 6)
        lon = np.round(target_lon + np.random.normal(0, 0.015), 6)
        
        # Clamp within Delhi-NCR bounding box
        lat = float(np.clip(lat, 28.4000, 28.8500))
        lon = float(np.clip(lon, 76.9000, 77.4500))
        
        hours_ago = np.random.uniform(0.5, 72.0)
        last_seen = base_date - timedelta(hours=hours_ago)
        day_str = last_seen.strftime("%A")
        weather = np.random.choice(weather_options)
        
        avg_dist = np.round(np.random.uniform(1.2, 18.5), 2)
        avg_speed = np.round(np.random.uniform(3.5, 45.0), 2)
        time_since_last_seen = np.round(hours_ago, 1)
        
        records.append({
            "Case_ID": case_id,
            "Person_ID": person_id,
            "Age_Group": age,
            "Last_Latitude": lat,
            "Last_Longitude": lon,
            "Last_Seen_Time": last_seen.strftime("%Y-%m-%d %H:%M:%S"),
            "Day": day_str,
            "Weather": weather,
            "Usual_Area": usual,
            "Average_Distance": avg_dist,
            "Average_Speed": avg_speed,
            "Previous_Area": prev,
            "Time_Since_Last_Seen": time_since_last_seen,
            "Target_Area": target
        })
        
    df = pd.DataFrame(records)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Synthetic Case Data saved to: {output_file} ({len(df)} records)")
    return df

if __name__ == "__main__":
    generate_fictional_cases()
