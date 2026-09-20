"""Forecast bust prediction and lead-time trajectory API routes."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from bustwatch.data.features import extract_atmospheric_features
from bustwatch.data.schemas import (
    BustPredictionOutput,
    LeadTimeForecast,
    StationMetadata,
)
from bustwatch.renderer.confidence_map import get_risk_level

router = APIRouter(prefix="/forecast", tags=["Forecast Bust Detection"])


SAMPLE_STATIONS: list[StationMetadata] = [
    StationMetadata(station_id="KORD", name="Chicago O'Hare, IL", latitude=41.9742, longitude=-87.9073, elevation_m=204.0, climate_zone="humid_continental"),
    StationMetadata(station_id="KDEN", name="Denver International, CO", latitude=39.8561, longitude=-104.6737, elevation_m=1656.0, climate_zone="semi_arid_leeward"),
    StationMetadata(station_id="KJFK", name="New York JFK, NY", latitude=40.6413, longitude=-73.7781, elevation_m=4.0, climate_zone="coastal_maritime"),
    StationMetadata(station_id="KSEA", name="Seattle Tacoma, WA", latitude=47.4502, longitude=-122.3088, elevation_m=132.0, climate_zone="pacific_northwest_orographic"),
    StationMetadata(station_id="KMCO", name="Orlando International, FL", latitude=28.4312, longitude=-81.3081, elevation_m=29.0, climate_zone="subtropical_convective"),
    StationMetadata(station_id="EDDF", name="Frankfurt Airport, Germany", latitude=50.0379, longitude=8.5622, elevation_m=111.0, climate_zone="western_europe_maritime"),
    StationMetadata(station_id="RJTT", name="Tokyo Haneda, Japan", latitude=35.5494, longitude=139.7798, elevation_m=6.0, climate_zone="east_asia_monsoonal"),
]


class PredictBustRequest(BaseModel):
    """Input payload for point forecast bust prediction."""
    lead_time_days: int = Field(default=5, ge=1, le=10, description="Lead time in days (1–10)")
    latitude: float = Field(default=41.97, ge=-90.0, le=90.0)
    longitude: float = Field(default=-87.90, ge=-180.0, le=180.0)
    ensemble_spread_z500: float = Field(default=45.0, ge=0.0, description="500hPa ensemble standard deviation (m)")
    ensemble_spread_t2m: float = Field(default=2.8, ge=0.0, description="2m temperature ensemble std (°C)")
    ens_mean_det_diff_z500: float = Field(default=12.0, ge=0.0, description="Ensemble mean minus deterministic height (m)")
    rossby_wave_activity: float = Field(default=14.5, ge=0.0, description="Rossby wave packet envelope / v'2 (m2/s2)")
    blocking_metric: float = Field(default=0.8, description="Tibaldi-Molteni blocking gradient index")
    teleconnection_pna: float = Field(default=0.4, description="PNA index proxy")
    teleconnection_nao: float = Field(default=-0.3, description="NAO index proxy")
    zonal_wind_shear_250_850: float = Field(default=18.0, description="250hPa - 850hPa wind shear (m/s)")
    climatological_temp_anomaly: float = Field(default=1.5, description="Temperature anomaly vs climatology (°C)")


@router.get("/stations", response_model=list[StationMetadata])
async def list_sample_stations():
    """Retrieve curated list of reference meteorological verification stations."""
    return SAMPLE_STATIONS


@router.post("/bust-risk", response_model=BustPredictionOutput)
async def predict_bust_risk(request: Request, payload: PredictBustRequest):
    """
    Predict calibrated probability of a forecast bust and expected error bounds.
    """
    classifier = getattr(request.app.state, "classifier", None)
    regressor = getattr(request.app.state, "regressor", None)

    # Seasonal harmonics for current date
    now = datetime.now(timezone.utc)
    doy = now.timetuple().tm_yday
    doy_rad = (2.0 * np.pi * (doy - 1)) / 365.25

    feat_dict = {
        "lead_time_days": float(payload.lead_time_days),
        "latitude": float(payload.latitude),
        "longitude": float(payload.longitude),
        "ensemble_spread_z500": float(payload.ensemble_spread_z500),
        "ensemble_spread_t2m": float(payload.ensemble_spread_t2m),
        "ens_mean_det_diff_z500": float(payload.ens_mean_det_diff_z500),
        "rossby_wave_activity": float(payload.rossby_wave_activity),
        "blocking_metric": float(payload.blocking_metric),
        "teleconnection_pna": float(payload.teleconnection_pna),
        "teleconnection_nao": float(payload.teleconnection_nao),
        "zonal_wind_shear_250_850": float(payload.zonal_wind_shear_250_850),
        "climatological_temp_anomaly": float(payload.climatological_temp_anomaly),
        "day_of_year_sin": float(np.sin(doy_rad)),
        "day_of_year_cos": float(np.cos(doy_rad)),
    }

    x_vec = extract_atmospheric_features(feat_dict).reshape(1, -1)

    if classifier and classifier.is_fitted:
        raw_p = float(classifier.predict_proba_raw(x_vec)[0])
        cal_p = float(classifier.predict_proba(x_vec)[0])
        feat_importances = classifier.get_feature_importances()
    else:
        # Physical proxy
        logit = -3.5 + 0.35 * payload.lead_time_days + 0.03 * payload.ensemble_spread_z500 + 0.4 * payload.blocking_metric
        cal_p = float(1.0 / (1.0 + np.exp(-logit)))
        raw_p = cal_p
        feat_importances = {"lead_time_days": 0.35, "ensemble_spread_z500": 0.25, "blocking_metric": 0.20}

    if regressor and regressor.is_fitted:
        err_res = regressor.predict_error_quantiles(x_vec)
        q10_z = float(err_res["z500_error_m"][0.10][0])
        q50_z = float(err_res["z500_error_m"][0.50][0])
        q90_z = float(err_res["z500_error_m"][0.90][0])
        q50_t = float(err_res["t2m_error_c"][0.50][0])
    else:
        q50_z = float(10.0 * np.exp(0.18 * payload.lead_time_days) + payload.ensemble_spread_z500 * 0.4)
        q10_z = float(q50_z * 0.45)
        q90_z = float(q50_z * 1.85)
        q50_t = float(1.2 * np.exp(0.12 * payload.lead_time_days) + payload.ensemble_spread_t2m * 0.3)

    risk_level = get_risk_level(cal_p)

    return BustPredictionOutput(
        bust_probability_raw=round(raw_p, 4),
        bust_probability_calibrated=round(cal_p, 4),
        is_bust_predicted=bool(cal_p >= 0.5),
        confidence_score=round(1.0 - cal_p, 4),
        expected_error_z500_m=round(q50_z, 2),
        expected_error_t2m_c=round(q50_t, 2),
        confidence_interval_z500_90=(round(q10_z, 2), round(q90_z, 2)),
        risk_level=risk_level,
        contributing_factors={k: round(v, 4) for k, v in feat_importances.items()},
    )


@router.get("/lead-profile", response_model=LeadTimeForecast)
async def get_lead_profile(
    request: Request,
    latitude: float = Query(default=41.97, ge=-90.0, le=90.0),
    longitude: float = Query(default=-87.90, ge=-180.0, le=180.0),
):
    """Retrieve full Day 1–10 forecast bust risk profile for specific coordinates."""
    renderer = getattr(request.app.state, "renderer", None)
    if renderer is None:
        from bustwatch.renderer.confidence_map import ConfidenceMapRenderer
        renderer = ConfidenceMapRenderer(
            classifier=getattr(request.app.state, "classifier", None),
            regressor=getattr(request.app.state, "regressor", None),
        )

    return renderer.compute_location_lead_profile(latitude, longitude)
