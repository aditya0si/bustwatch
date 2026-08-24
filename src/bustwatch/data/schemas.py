"""Pydantic schemas and dataclasses for meteorological forecast and observation records."""

from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


class SpatialBoundingBox(BaseModel):
    """Bounding box for geospatial queries."""
    lat_min: float = Field(default=20.0, ge=-90.0, le=90.0, description="Minimum latitude (degrees)")
    lat_max: float = Field(default=60.0, ge=-90.0, le=90.0, description="Maximum latitude (degrees)")
    lon_min: float = Field(default=-130.0, ge=-180.0, le=180.0, description="Minimum longitude (degrees)")
    lon_max: float = Field(default=-60.0, ge=-180.0, le=180.0, description="Maximum longitude (degrees)")


class StationMetadata(BaseModel):
    """Station or grid point identification metadata."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float = 0.0
    climate_zone: str = "temperate"


class GridPoint(BaseModel):
    """Geospatial point with meteorological state."""
    latitude: float
    longitude: float
    lead_time_days: int
    deterministic_z500: float  # Geopotential height at 500hPa (gpm)
    deterministic_t2m: float   # 2-meter temperature (°C)
    deterministic_u250: float  # 250hPa zonal wind (m/s)
    deterministic_v250: float  # 250hPa meridional wind (m/s)
    ensemble_z500_mean: float
    ensemble_z500_std: float   # Ensemble spread
    ensemble_t2m_mean: float
    ensemble_t2m_std: float
    z500_anomaly: float = 0.0  # Anomaly relative to climatology
    blocking_metric: float = 0.0 # Tibaldi-Molteni gradient proxy


class ForecastObservationPair(BaseModel):
    """Paired NWP forecast and verifying reanalysis/observation record."""
    record_id: str
    forecast_init_time: datetime
    valid_time: datetime
    lead_time_days: int
    latitude: float
    longitude: float
    forecast_z500: float
    forecast_t2m: float
    observed_z500: float
    observed_t2m: float
    ensemble_spread_z500: float
    ensemble_spread_t2m: float
    rossby_wave_activity: float
    blocking_metric: float
    teleconnection_pna: float = 0.0
    teleconnection_nao: float = 0.0
    error_z500: float = 0.0
    error_t2m: float = 0.0
    is_bust: bool = False


class AtmosphericFeatures(BaseModel):
    """Engineered input feature vector for ML model inference."""
    lead_time_days: float
    latitude: float
    longitude: float
    ensemble_spread_z500: float
    ensemble_spread_t2m: float
    ens_mean_det_diff_z500: float
    rossby_wave_activity: float
    blocking_metric: float
    teleconnection_pna: float
    teleconnection_nao: float
    zonal_wind_shear_250_850: float
    climatological_temp_anomaly: float
    day_of_year_sin: float
    day_of_year_cos: float


class BustPredictionOutput(BaseModel):
    """Inference output for forecast bust prediction."""
    bust_probability_raw: float = Field(ge=0.0, le=1.0)
    bust_probability_calibrated: float = Field(ge=0.0, le=1.0)
    is_bust_predicted: bool
    confidence_score: float = Field(ge=0.0, le=1.0, description="1.0 - calibrated_bust_probability")
    expected_error_z500_m: float
    expected_error_t2m_c: float
    confidence_interval_z500_90: Tuple[float, float]
    risk_level: str  # "LOW", "MODERATE", "ELEVATED", "CRITICAL_BUST"
    contributing_factors: Dict[str, float]


class LeadTimeForecast(BaseModel):
    """Complete 10-day lead-time profile for a location."""
    latitude: float
    longitude: float
    init_time: datetime
    days: List[int]
    bust_probabilities: List[float]
    confidence_scores: List[float]
    expected_errors_z500: List[float]
    risk_levels: List[str]


class BustRecord(BaseModel):
    """Historical verified bust event log."""
    event_id: str
    date: datetime
    lead_time_days: int
    location_name: str
    latitude: float
    longitude: float
    bust_type: str  # "BLOCKING_COLLAPSE", "ROSSBY_WAVE_BREAKING", "CONVECTIVE_FEEDBACK"
    observed_error_z500: float
    calibrated_bust_prob: float
    description: str
