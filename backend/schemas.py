"""
Pydantic schemas for CASEFILE API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime


class CaseInput(BaseModel):
    """Input schema for creating or analyzing a fictional case."""
    Case_ID: Optional[str] = None
    Person_ID: Optional[str] = None
    Age_Group: str = Field(..., description="Age group: Child (5-12), Teen (13-17), Young Adult (18-35), Adult (36-60), Senior (60+)")
    Last_Latitude: float = Field(..., ge=-90, le=90, description="Last known latitude")
    Last_Longitude: float = Field(..., ge=-180, le=180, description="Last known longitude")
    Last_Seen_Time: str = Field(..., description="Last seen timestamp (YYYY-MM-DD HH:MM:SS)")
    Day: str = Field(..., description="Day of week")
    Weather: str = Field(default="Clear", description="Weather condition")
    Usual_Area: str = Field(..., description="Person's usual area")
    Average_Distance: float = Field(default=5.0, ge=0, description="Average travel distance in km")
    Average_Speed: float = Field(default=15.0, ge=0, description="Average movement speed in km/h")
    Previous_Area: str = Field(..., description="Previously visited area")
    Time_Since_Last_Seen: float = Field(..., ge=0, description="Hours since last observation")

    @field_validator("Last_Latitude")
    @classmethod
    def validate_latitude(cls, v):
        if v < -90 or v > 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @field_validator("Last_Longitude")
    @classmethod
    def validate_longitude(cls, v):
        if v < -180 or v > 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v

    @field_validator("Last_Seen_Time")
    @classmethod
    def validate_timestamp(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("Timestamp must be in format YYYY-MM-DD HH:MM:SS")
        return v


class FrequentLocation(BaseModel):
    grid_cell_id: str
    visit_count: int
    avg_lat: float
    avg_lon: float
    avg_speed: float
    total_time_min: float


class AnomalyPoint(BaseModel):
    latitude: float
    longitude: float
    anomaly_score: float
    speed: float
    distance: float


class ProbableArea(BaseModel):
    area_name: str
    ml_probability: float
    visit_frequency: float
    route_similarity: float
    distance_relevance: float
    time_relevance: float
    anomaly_evidence: float
    priority_score: float
    priority_level: str
    area_latitude: float
    area_longitude: float


class RouteStep(BaseModel):
    path: List[str]
    probability: float


class ExplanationFactor(BaseModel):
    factor: str
    detail: str
    weight: str
    contribution: str


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class AreaExplanation(BaseModel):
    area_name: str
    priority_score: float
    priority_level: str
    ml_probability: float
    summary: str
    supporting_factors: List[ExplanationFactor]
    top_model_features: List[FeatureImportance]


class MovementPattern(BaseModel):
    user_id: str
    total_distance_m: float
    avg_speed_kmh: float
    max_speed_kmh: float
    duration_minutes: float
    locations_visited_count: int


class ClusterInfo(BaseModel):
    cluster_id: int
    center_lat: float
    center_lon: float
    point_count: int
    avg_speed: float


class AnalysisResponse(BaseModel):
    """Complete analysis results returned to the frontend."""
    case_id: str
    status: str
    frequent_locations: List[FrequentLocation]
    movement_patterns: List[MovementPattern]
    clusters: List[ClusterInfo]
    anomalies: List[AnomalyPoint]
    probable_areas: List[ProbableArea]
    predicted_routes: List[RouteStep]
    explanations: List[AreaExplanation]
    summary: str


class CaseRecord(BaseModel):
    """Stored case record."""
    Case_ID: str
    Person_ID: str
    Age_Group: str
    Last_Latitude: float
    Last_Longitude: float
    Last_Seen_Time: str
    Day: str
    Weather: str
    Usual_Area: str
    Average_Distance: float
    Average_Speed: float
    Previous_Area: str
    Time_Since_Last_Seen: float
    Target_Area: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
