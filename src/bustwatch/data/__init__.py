"""Data pipeline, schemas, connectors, and feature engineering for BustWatch."""

from bustwatch.data.schemas import (
    ForecastObservationPair,
    BustRecord,
    AtmosphericFeatures,
    GridPoint,
    LeadTimeForecast,
    StationMetadata,
    SpatialBoundingBox,
)
from bustwatch.data.features import extract_atmospheric_features, compute_bust_labels
from bustwatch.data.synthetic import generate_synthetic_dataset, generate_spatial_grid_forecast
from bustwatch.data.noaa_connector import NOAADataConnector
from bustwatch.data.openmeteo_connector import OpenMeteoConnector

__all__ = [
    "ForecastObservationPair",
    "BustRecord",
    "AtmosphericFeatures",
    "GridPoint",
    "LeadTimeForecast",
    "StationMetadata",
    "SpatialBoundingBox",
    "extract_atmospheric_features",
    "compute_bust_labels",
    "generate_synthetic_dataset",
    "generate_spatial_grid_forecast",
    "NOAADataConnector",
    "OpenMeteoConnector",
]
