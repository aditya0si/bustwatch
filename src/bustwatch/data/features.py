"""Atmospheric feature extraction and bust label computation."""

from __future__ import annotations
import math
from datetime import datetime
from typing import Dict, List, Any, Union
import numpy as np
import pandas as pd

from bustwatch.config import settings
from bustwatch.data.schemas import AtmosphericFeatures, ForecastObservationPair


FEATURE_COLUMNS = [
    "lead_time_days",
    "latitude",
    "longitude",
    "ensemble_spread_z500",
    "ensemble_spread_t2m",
    "ens_mean_det_diff_z500",
    "rossby_wave_activity",
    "blocking_metric",
    "teleconnection_pna",
    "teleconnection_nao",
    "zonal_wind_shear_250_850",
    "climatological_temp_anomaly",
    "day_of_year_sin",
    "day_of_year_cos",
]


def compute_seasonal_harmonics(dt: datetime) -> tuple[float, float]:
    """Compute sinusoidal day-of-year seasonal cyclical features."""
    doy = dt.timetuple().tm_yday
    rad = (2.0 * math.pi * (doy - 1)) / 365.25
    return float(np.sin(rad)), float(np.cos(rad))


def compute_blocking_index(
    z500_north: float, z500_mid: float, z500_south: float, delta_lat_deg: float = 20.0
) -> float:
    """
    Compute Tibaldi-Molteni style geopotential height gradient index (GHGS proxy).
    GHGS = (Z(lat_mid) - Z(lat_south)) / delta_lat
    A negative or strongly reversed gradient indicates upper-level atmospheric blocking.
    """
    ghgs = (z500_mid - z500_south) / delta_lat_deg
    ghgn = (z500_north - z500_mid) / delta_lat_deg
    # Positive index signifies anticyclonic ridge / blocking signature
    return float((z500_mid - (z500_north + z500_south) / 2.0) / 10.0)


def extract_atmospheric_features(
    record: Union[ForecastObservationPair, Dict[str, Any], pd.Series]
) -> np.ndarray:
    """Extract ordered 1D numpy array of atmospheric features from record."""
    if isinstance(record, ForecastObservationPair):
        dt = record.valid_time
        doy_sin, doy_cos = compute_seasonal_harmonics(dt)
        ens_diff = abs(record.forecast_z500 - record.forecast_z500) # baseline
        features = [
            float(record.lead_time_days),
            float(record.latitude),
            float(record.longitude),
            float(record.ensemble_spread_z500),
            float(record.ensemble_spread_t2m),
            float(ens_diff),
            float(record.rossby_wave_activity),
            float(record.blocking_metric),
            float(record.teleconnection_pna),
            float(record.teleconnection_nao),
            float(record.rossby_wave_activity * 0.75),  # shear proxy
            float(record.forecast_t2m - 15.0),          # temp anomaly proxy
            float(doy_sin),
            float(doy_cos),
        ]
    elif isinstance(record, (dict, pd.Series)):
        features = [float(record[col]) for col in FEATURE_COLUMNS]
    else:
        raise TypeError(f"Unsupported record type: {type(record)}")

    return np.array(features, dtype=np.float32)


def compute_bust_labels(
    df: pd.DataFrame,
    threshold_z500: float = settings.bust_threshold_z500_m,
    threshold_t2m: float = settings.bust_threshold_t2m_c,
    use_percentile: bool = False,
    percentile: float = settings.bust_percentile_threshold,
) -> pd.DataFrame:
    """
    Compute absolute errors and ground-truth binary bust labels.
    A bust occurs when forecast error exceeds operational threshold or regional 90th percentile.
    """
    df = df.copy()
    if "error_z500" not in df.columns:
        df["error_z500"] = (df["forecast_z500"] - df["observed_z500"]).abs()
    if "error_t2m" not in df.columns:
        df["error_t2m"] = (df["forecast_t2m"] - df["observed_t2m"]).abs()

    if use_percentile:
        p_z500 = np.percentile(df["error_z500"], percentile)
        p_t2m = np.percentile(df["error_t2m"], percentile)
        df["is_bust"] = ((df["error_z500"] >= p_z500) | (df["error_t2m"] >= p_t2m)).astype(int)
    else:
        df["is_bust"] = (
            (df["error_z500"] >= threshold_z500) | (df["error_t2m"] >= threshold_t2m)
        ).astype(int)

    return df
