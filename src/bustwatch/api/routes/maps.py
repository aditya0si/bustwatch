"""Spatial 2D Confidence Grid and GeoJSON map routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from bustwatch.data.schemas import SpatialBoundingBox
from bustwatch.renderer.confidence_map import ConfidenceMapRenderer

router = APIRouter(prefix="/maps", tags=["Confidence Maps"])


@router.get("/confidence-grid")
async def get_confidence_grid(
    request: Request,
    lead_time_days: int = Query(default=5, ge=1, le=10, description="Forecast lead time in days (1–10)"),
    lat_min: float = Query(default=24.0, ge=-90.0, le=90.0),
    lat_max: float = Query(default=50.0, ge=-90.0, le=90.0),
    lon_min: float = Query(default=-125.0, ge=-180.0, le=180.0),
    lon_max: float = Query(default=-65.0, ge=-180.0, le=180.0),
    resolution_deg: float = Query(default=2.0, ge=0.5, le=5.0),
) -> dict[str, Any]:
    """
    Retrieve dense 2D rasters of calibrated bust probability, confidence score, and expected error.
    """
    renderer: ConfidenceMapRenderer = getattr(request.app.state, "renderer", None)
    if renderer is None:
        renderer = ConfidenceMapRenderer(
            classifier=getattr(request.app.state, "classifier", None),
            regressor=getattr(request.app.state, "regressor", None),
        )

    bbox = SpatialBoundingBox(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    return renderer.render_grid_raster(
        lead_time_days=lead_time_days,
        bbox=bbox,
        resolution_deg=resolution_deg,
    )


@router.get("/geojson")
async def get_geojson_layer(
    request: Request,
    lead_time_days: int = Query(default=5, ge=1, le=10),
    lat_min: float = Query(default=24.0, ge=-90.0, le=90.0),
    lat_max: float = Query(default=50.0, ge=-90.0, le=90.0),
    lon_min: float = Query(default=-125.0, ge=-180.0, le=180.0),
    lon_max: float = Query(default=-65.0, ge=-180.0, le=180.0),
    resolution_deg: float = Query(default=2.5, ge=0.5, le=5.0),
) -> dict[str, Any]:
    """
    Retrieve GeoJSON FeatureCollection containing colored risk polygon cells for mapping.
    """
    renderer: ConfidenceMapRenderer = getattr(request.app.state, "renderer", None)
    if renderer is None:
        renderer = ConfidenceMapRenderer(
            classifier=getattr(request.app.state, "classifier", None),
            regressor=getattr(request.app.state, "regressor", None),
        )

    bbox = SpatialBoundingBox(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    return renderer.render_geojson_contours(
        lead_time_days=lead_time_days,
        bbox=bbox,
        resolution_deg=resolution_deg,
    )
