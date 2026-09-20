"""Unit tests for meteorological data structures, feature extraction, and connectors."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bustwatch.data.features import (
    FEATURE_COLUMNS,
    compute_blocking_index,
    compute_bust_labels,
    compute_seasonal_harmonics,
    extract_atmospheric_features,
)
from bustwatch.data.noaa_connector import NOAADataConnector
from bustwatch.data.openmeteo_connector import OpenMeteoConnector
from bustwatch.data.schemas import (
    ForecastObservationPair,
    GridPoint,
    SpatialBoundingBox,
)
from bustwatch.data.synthetic import (
    generate_spatial_grid_forecast,
    generate_synthetic_dataset,
    generate_synthetic_grid_point,
)


def test_seasonal_harmonics():
    """Verify seasonal sinusoidal cyclical encoding."""
    dt_summer = datetime(2025, 6, 21, tzinfo=timezone.utc)
    sin_val, cos_val = compute_seasonal_harmonics(dt_summer)
    assert -1.0 <= sin_val <= 1.0
    assert -1.0 <= cos_val <= 1.0
    assert np.isclose(sin_val**2 + cos_val**2, 1.0, atol=1e-3)


def test_blocking_index():
    """Verify Tibaldi-Molteni blocking gradient proxy calculation."""
    idx = compute_blocking_index(z500_north=5300.0, z500_mid=5700.0, z500_south=5600.0)
    assert isinstance(idx, float)
    assert idx > 0.0  # Positive ridge anomaly


def test_synthetic_grid_point():
    """Verify synthetic meteorological point generator."""
    pt = generate_synthetic_grid_point(latitude=42.0, longitude=-88.0, lead_time_days=5, seed=42)
    assert isinstance(pt, GridPoint)
    assert pt.latitude == 42.0
    assert pt.longitude == -88.0
    assert pt.lead_time_days == 5
    assert 4800.0 <= pt.deterministic_z500 <= 6000.0
    assert pt.ensemble_z500_std > 0.0


def test_synthetic_dataset():
    """Verify synthetic dataset generator produces valid DataFrame."""
    df = generate_synthetic_dataset(n_samples=50, random_seed=42)
    assert len(df) == 50
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert "error_z500" in df.columns
    assert "error_t2m" in df.columns
    assert "is_bust" in df.columns


def test_compute_bust_labels():
    """Verify bust labeling logic based on threshold and percentile."""
    df = pd.DataFrame({
        "forecast_z500": [5600.0, 5600.0, 5600.0],
        "observed_z500": [5610.0, 5680.0, 5700.0],
        "forecast_t2m": [15.0, 15.0, 20.0],
        "observed_t2m": [15.5, 16.0, 15.0],
    })
    labeled = compute_bust_labels(df, threshold_z500=60.0, threshold_t2m=3.5)
    assert labeled["is_bust"].tolist() == [0, 1, 1]


def test_extract_atmospheric_features_record():
    """Verify feature vector extraction from ForecastObservationPair."""
    pair = ForecastObservationPair(
        record_id="rec_001",
        forecast_init_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        valid_time=datetime(2025, 1, 6, tzinfo=timezone.utc),
        lead_time_days=5,
        latitude=40.0,
        longitude=-90.0,
        forecast_z500=5500.0,
        forecast_t2m=12.0,
        observed_z500=5580.0,
        observed_t2m=14.0,
        ensemble_spread_z500=35.0,
        ensemble_spread_t2m=2.1,
        rossby_wave_activity=12.0,
        blocking_metric=0.5,
        error_z500=80.0,
        error_t2m=2.0,
        is_bust=True,
    )
    vec = extract_atmospheric_features(pair)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (len(FEATURE_COLUMNS),)
    assert vec[0] == 5.0  # lead_time_days


def test_spatial_grid_forecast():
    """Verify 2D spatial grid generator."""
    bbox = SpatialBoundingBox(lat_min=30.0, lat_max=40.0, lon_min=-100.0, lon_max=-90.0)
    grid = generate_spatial_grid_forecast(lead_time_days=3, bbox=bbox, resolution_deg=5.0)
    assert grid["lead_time_days"] == 3
    assert len(grid["latitudes"]) == 3
    assert len(grid["longitudes"]) == 3
    assert len(grid["grid"]) == 3


def test_noaa_connector_urls():
    """Verify NOAA AWS S3 URL generator."""
    conn = NOAADataConnector()
    init_time = datetime(2025, 5, 10, 12, 0, tzinfo=timezone.utc)
    url = conn.build_gfs_url(init_time, forecast_hour=120)
    assert "gfs.20250510/12/atmos/gfs.t12z.pgrb2.0p25.f120" in url
    idx_url = conn.build_index_url(init_time, forecast_hour=120)
    assert idx_url.endswith(".idx")


def test_openmeteo_connector_fallback():
    """Verify OpenMeteo connector fallback grid point generation."""
    conn = OpenMeteoConnector(timeout=1)
    pt = conn.get_grid_point_forecast(latitude=45.0, longitude=-75.0, lead_time_days=4)
    assert isinstance(pt, GridPoint)
    assert pt.latitude == 45.0
    assert pt.longitude == -75.0
