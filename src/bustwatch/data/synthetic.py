"""Physically consistent synthetic meteorological error & forecast generator for BustWatch."""

from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from bustwatch.config import settings
from bustwatch.data.schemas import GridPoint, ForecastObservationPair, SpatialBoundingBox
from bustwatch.data.features import compute_seasonal_harmonics, FEATURE_COLUMNS


def generate_synthetic_grid_point(
    latitude: float,
    longitude: float,
    lead_time_days: int = 5,
    seed: Optional[int] = None,
) -> GridPoint:
    """Generate a single physically consistent synthetic meteorological grid point."""
    if seed is not None:
        np.random.seed(seed)

    # Physical baseline gradients
    lat_factor = (latitude - 40.0) / 30.0  # -1 at 10N, +1 at 70N
    z500_base = 5600.0 - (lat_factor * 450.0)
    t2m_base = 15.0 - (lat_factor * 30.0)

    # Rossby wave perturbations
    wave_num_4 = np.sin(np.radians(longitude * 4.0)) * 60.0
    wave_num_6 = np.cos(np.radians(longitude * 6.0)) * 30.0
    z500_syn = z500_base + wave_num_4 + wave_num_6

    # Error growth with lead time (Lyapunov exponent ~ 0.25 day^-1)
    spread_z500 = 10.0 * np.exp(0.24 * lead_time_days) + np.random.uniform(2.0, 8.0)
    spread_t2m = 0.8 * np.exp(0.20 * lead_time_days) + np.random.uniform(0.1, 0.5)

    # Blocking condition
    blocking_metric = float(np.clip(np.sin(np.radians(longitude * 2.0)) * 1.5 + np.random.normal(0, 0.4), -2.0, 3.0))

    return GridPoint(
        latitude=latitude,
        longitude=longitude,
        lead_time_days=lead_time_days,
        deterministic_z500=float(z500_syn),
        deterministic_t2m=float(t2m_base + np.random.normal(0, 1.5)),
        deterministic_u250=float(30.0 + np.random.uniform(-10, 20)),
        deterministic_v250=float(np.random.normal(0, 8)),
        ensemble_z500_mean=float(z500_syn + np.random.normal(0, spread_z500 * 0.3)),
        ensemble_z500_std=float(spread_z500),
        ensemble_t2m_mean=float(t2m_base),
        ensemble_t2m_std=float(spread_t2m),
        z500_anomaly=float(wave_num_4),
        blocking_metric=blocking_metric,
    )


def generate_synthetic_dataset(
    n_samples: int = 1200,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate rich, physically-grounded synthetic training/testing dataset
    simulating NWP medium-range forecast errors and bust regimes.
    """
    rng = np.random.default_rng(random_seed)

    base_time = datetime(2025, 1, 1, 0, 0)
    records: List[Dict[str, Any]] = []

    for i in range(n_samples):
        # Sample lead time (1 to 10 days)
        lead_time = int(rng.integers(1, 11))
        # Sample geographic coordinates (focus on mid-latitudes Northern Hemisphere)
        lat = float(rng.uniform(25.0, 65.0))
        lon = float(rng.uniform(-135.0, -60.0))

        # Sample time across seasons
        days_offset = int(rng.integers(0, 365))
        valid_dt = base_time + timedelta(days=days_offset + lead_time)
        init_dt = base_time + timedelta(days=days_offset)
        doy_sin, doy_cos = compute_seasonal_harmonics(valid_dt)

        # Atmospheric regimes
        # 1. Rossby Wave Packet Activity (RWPA)
        rossby_wave_activity = float(rng.gamma(shape=2.5, scale=4.0))  # 0 to 30 m/s
        # 2. Blocking index (Tibaldi-Molteni gradient)
        blocking_metric = float(rng.normal(loc=-0.2, scale=1.2))
        # 3. Teleconnection indices (NAO & PNA)
        tele_pna = float(rng.normal(0.0, 1.0))
        tele_nao = float(rng.normal(0.0, 1.0))

        # Ensemble spread: baseline exponential growth with chaotic fluctuations
        base_spread_z500 = 8.0 * np.exp(0.25 * lead_time)
        spread_z500 = float(base_spread_z500 * rng.lognormal(mean=0.0, sigma=0.35))
        spread_t2m = float((1.0 + 0.5 * lead_time) * rng.lognormal(mean=0.0, sigma=0.25))

        # Divergence between ensemble mean and deterministic run
        ens_mean_det_diff = float(rng.exponential(scale=spread_z500 * 0.45))

        # Physics of Forecast Busts:
        # Bust probability increases strongly with:
        # - High lead time
        # - High Rossby wave activity & wave breaking
        # - Strong blocking regimes
        # - Large spread-skill divergence
        # - High PNA/NAO teleconnection phases
        logit_bust = (
            -4.5
            + 0.42 * lead_time
            + 0.045 * spread_z500
            + 0.08 * rossby_wave_activity
            + 0.65 * max(0.0, blocking_metric)
            + 0.35 * max(0.0, tele_pna)
            + 0.03 * ens_mean_det_diff
        )
        true_prob = 1.0 / (1.0 + np.exp(-logit_bust))

        # Ground truth observation vs forecast
        base_z500 = 5500.0 - (lat - 40.0) * 15.0 + np.sin(np.radians(lon * 3)) * 80.0
        base_t2m = 12.0 - (lat - 40.0) * 0.8 + 8.0 * doy_sin

        # Error generation
        is_bust_event = rng.uniform(0.0, 1.0) < true_prob
        if is_bust_event:
            # Extreme error regime
            error_z500 = float(rng.uniform(settings.bust_threshold_z500_m, 180.0) + 10.0 * lead_time * 0.5)
            error_t2m = float(rng.uniform(settings.bust_threshold_t2m_c, 9.5))
        else:
            # Standard sub-threshold error
            error_z500 = float(rng.gamma(shape=2.0, scale=min(12.0, 2.5 * lead_time)))
            error_t2m = float(rng.gamma(shape=2.0, scale=min(0.8, 0.25 * lead_time)))

        fcst_z500 = base_z500 + rng.normal(0, 10.0)
        obs_z500 = fcst_z500 + (error_z500 if rng.random() > 0.5 else -error_z500)

        fcst_t2m = base_t2m + rng.normal(0, 1.0)
        obs_t2m = fcst_t2m + (error_t2m if rng.random() > 0.5 else -error_t2m)

        shear = float(rossby_wave_activity * 0.8 + rng.normal(0, 2.0))
        temp_anom = float(fcst_t2m - base_t2m)

        records.append({
            "record_id": f"rec_{i:06d}",
            "forecast_init_time": init_dt,
            "valid_time": valid_dt,
            "lead_time_days": lead_time,
            "latitude": lat,
            "longitude": lon,
            "forecast_z500": fcst_z500,
            "forecast_t2m": fcst_t2m,
            "observed_z500": obs_z500,
            "observed_t2m": obs_t2m,
            "ensemble_spread_z500": spread_z500,
            "ensemble_spread_t2m": spread_t2m,
            "ens_mean_det_diff_z500": ens_mean_det_diff,
            "rossby_wave_activity": rossby_wave_activity,
            "blocking_metric": blocking_metric,
            "teleconnection_pna": tele_pna,
            "teleconnection_nao": tele_nao,
            "zonal_wind_shear_250_850": shear,
            "climatological_temp_anomaly": temp_anom,
            "day_of_year_sin": doy_sin,
            "day_of_year_cos": doy_cos,
            "error_z500": error_z500,
            "error_t2m": error_t2m,
            "is_bust": int(is_bust_event),
        })

    return pd.DataFrame(records)


def generate_spatial_grid_forecast(
    lead_time_days: int = 5,
    bbox: Optional[SpatialBoundingBox] = None,
    resolution_deg: float = 2.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a 2D spatial raster grid of forecast states and ensemble characteristics.
    """
    if bbox is None:
        bbox = SpatialBoundingBox()

    lats = np.arange(bbox.lat_min, bbox.lat_max + resolution_deg, resolution_deg)
    lons = np.arange(bbox.lon_min, bbox.lon_max + resolution_deg, resolution_deg)

    grid_points = []
    for i, lat in enumerate(lats):
        row_points = []
        for j, lon in enumerate(lons):
            point_seed = int(seed + (i * 100) + j + lead_time_days * 1000)
            pt = generate_synthetic_grid_point(float(lat), float(lon), lead_time_days, seed=point_seed)
            row_points.append(pt.model_dump())
        grid_points.append(row_points)

    return {
        "lead_time_days": lead_time_days,
        "bbox": bbox.model_dump(),
        "resolution_deg": resolution_deg,
        "latitudes": [float(lat) for lat in lats],
        "longitudes": [float(lon) for lon in lons],
        "shape": [len(lats), len(lons)],
        "grid": grid_points,
    }
