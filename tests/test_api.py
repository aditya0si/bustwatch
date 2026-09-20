"""Integration tests for FastAPI REST API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from bustwatch.api.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Verify /health returns 200 and valid status."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["app_name"] == "BustWatch"
        assert "models" in data


@pytest.mark.asyncio
async def test_root_endpoint():
    """Verify root / returns 200 and docs/endpoints."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data
        assert "docs" in data


@pytest.mark.asyncio
async def test_list_stations():
    """Verify /api/v1/forecast/stations returns reference station list."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/forecast/stations")
        assert resp.status_code == 200
        stations = resp.json()
        assert isinstance(stations, list)
        assert len(stations) > 0
        assert "station_id" in stations[0]


@pytest.mark.asyncio
async def test_predict_bust_risk():
    """Verify /api/v1/forecast/bust-risk prediction endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        payload = {
            "lead_time_days": 6,
            "latitude": 41.97,
            "longitude": -87.90,
            "ensemble_spread_z500": 55.0,
            "ensemble_spread_t2m": 3.0,
            "ens_mean_det_diff_z500": 15.0,
            "rossby_wave_activity": 16.0,
            "blocking_metric": 1.2,
            "teleconnection_pna": 0.5,
            "teleconnection_nao": -0.2,
            "zonal_wind_shear_250_850": 18.0,
            "climatological_temp_anomaly": 2.0,
        }
        resp = await client.post("/api/v1/forecast/bust-risk", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "bust_probability_calibrated" in data
        assert "confidence_score" in data
        assert "expected_error_z500_m" in data
        assert "risk_level" in data
        assert 0.0 <= data["bust_probability_calibrated"] <= 1.0


@pytest.mark.asyncio
async def test_lead_profile_endpoint():
    """Verify /api/v1/forecast/lead-profile returns 10-day profile."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/forecast/lead-profile?latitude=40.0&longitude=-90.0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["days"]) == 10
        assert len(data["bust_probabilities"]) == 10


@pytest.mark.asyncio
async def test_confidence_grid_endpoint():
    """Verify /api/v1/maps/confidence-grid returns 2D matrices."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/maps/confidence-grid?lead_time_days=4&resolution_deg=5.0")
        assert resp.status_code == 200
        data = resp.json()
        assert "calibrated_bust_probability" in data
        assert "shape" in data


@pytest.mark.asyncio
async def test_geojson_endpoint():
    """Verify /api/v1/maps/geojson returns FeatureCollection."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/maps/geojson?lead_time_days=3&resolution_deg=5.0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0


@pytest.mark.asyncio
async def test_eval_metrics_endpoint():
    """Verify /api/v1/evals/metrics returns verification benchmark payload."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/evals/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics_summary" in data
        assert "uncalibrated" in data["metrics_summary"]
        assert "calibrated" in data["metrics_summary"]


@pytest.mark.asyncio
async def test_historical_busts_endpoint():
    """Verify /api/v1/evals/historical-busts returns case studies."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/evals/historical-busts")
        assert resp.status_code == 200
        busts = resp.json()
        assert len(busts) >= 3
        assert busts[0]["event_id"].startswith("bust_")
