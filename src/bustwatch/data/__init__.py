"""Data pipeline, schemas, connectors, and feature engineering for BustWatch."""

from bustwatch.data.features import compute_bust_labels, extract_atmospheric_features
from bustwatch.data.noaa_connector import NOAADataConnector
from bustwatch.data.openmeteo_connector import OpenMeteoConnector
from bustwatch.data.schemas import (
    AtmosphericFeatures,
    BustRecord,
    ForecastObservationPair,
    GridPoint,
    LeadTimeForecast,
    SpatialBoundingBox,
    StationMetadata,
)
from bustwatch.data.synthetic import generate_spatial_grid_forecast, generate_synthetic_dataset

__all__ = [
    "AtmosphericFeatures",
    "BustRecord",
    "ForecastObservationPair",
    "GridPoint",
    "LeadTimeForecast",
    "NOAADataConnector",
    "OpenMeteoConnector",
    "SpatialBoundingBox",
    "StationMetadata",
    "compute_bust_labels",
    "extract_atmospheric_features",
    "generate_spatial_grid_forecast",
    "generate_synthetic_dataset",
]
