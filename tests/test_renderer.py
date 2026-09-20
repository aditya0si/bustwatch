"""Unit tests for ConfidenceMapRenderer and geospatial raster/vector outputs."""


from bustwatch.data.schemas import SpatialBoundingBox
from bustwatch.renderer.confidence_map import (
    ConfidenceMapRenderer,
    get_risk_color,
    get_risk_level,
)


def test_risk_level_classification():
    """Verify operational risk categorization thresholds."""
    assert get_risk_level(0.05) == "LOW"
    assert get_risk_level(0.20) == "MODERATE"
    assert get_risk_level(0.45) == "ELEVATED"
    assert get_risk_level(0.75) == "CRITICAL_BUST"


def test_risk_colors():
    """Verify hex color codes for each risk category."""
    assert get_risk_color("LOW").startswith("#")
    assert get_risk_color("CRITICAL_BUST").startswith("#")


def test_renderer_grid_raster():
    """Verify 2D spatial raster generation."""
    renderer = ConfidenceMapRenderer()
    bbox = SpatialBoundingBox(lat_min=30.0, lat_max=40.0, lon_min=-100.0, lon_max=-90.0)
    raster = renderer.render_grid_raster(lead_time_days=4, bbox=bbox, resolution_deg=5.0)

    assert raster["lead_time_days"] == 4
    assert "calibrated_bust_probability" in raster
    assert "confidence_score" in raster
    assert "expected_error_z500_m" in raster

    probs = raster["calibrated_bust_probability"]
    assert len(probs) == len(raster["latitudes"])
    assert len(probs[0]) == len(raster["longitudes"])


def test_renderer_geojson_contours():
    """Verify GeoJSON FeatureCollection generation."""
    renderer = ConfidenceMapRenderer()
    bbox = SpatialBoundingBox(lat_min=30.0, lat_max=35.0, lon_min=-95.0, lon_max=-90.0)
    geojson_out = renderer.render_geojson_contours(lead_time_days=5, bbox=bbox, resolution_deg=5.0)

    assert geojson_out["type"] == "FeatureCollection"
    assert len(geojson_out["features"]) > 0
    first_feat = geojson_out["features"][0]
    assert "bust_probability" in first_feat["properties"]
    assert "risk_level" in first_feat["properties"]
    assert "fill_color" in first_feat["properties"]


def test_renderer_location_lead_profile():
    """Verify 10-day lead time degradation curve computation."""
    renderer = ConfidenceMapRenderer()
    profile = renderer.compute_location_lead_profile(latitude=41.97, longitude=-87.90)

    assert len(profile.days) == 10
    assert len(profile.bust_probabilities) == 10
    assert len(profile.confidence_scores) == 10
    assert len(profile.expected_errors_z500) == 10
    assert len(profile.risk_levels) == 10
    # Expected error should generally grow with lead time
    assert profile.expected_errors_z500[-1] > profile.expected_errors_z500[0]
