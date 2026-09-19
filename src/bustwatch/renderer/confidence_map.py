"""Spatio-Temporal Confidence Map Renderer for Days 1–10 NWP Forecasts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import geojson
import numpy as np

from bustwatch.config import settings
from bustwatch.data.features import extract_atmospheric_features
from bustwatch.data.schemas import (
    GridPoint,
    LeadTimeForecast,
    SpatialBoundingBox,
)
from bustwatch.data.synthetic import generate_spatial_grid_forecast, generate_synthetic_grid_point
from bustwatch.models.bust_classifier import ForecastBustClassifier
from bustwatch.models.error_regressor import QuantileErrorRegressor


def get_risk_level(prob: float) -> str:
    """Classify calibrated bust probability into operational risk levels."""
    if prob < 0.15:
        return "LOW"
    elif prob < 0.35:
        return "MODERATE"
    elif prob < 0.60:
        return "ELEVATED"
    else:
        return "CRITICAL_BUST"


def get_risk_color(risk_level: str) -> str:
    """Hex color code corresponding to risk level."""
    mapping = {
        "LOW": "#10b981",         # Emerald-500
        "MODERATE": "#f59e0b",    # Amber-500
        "ELEVATED": "#f97316",    # Orange-500
        "CRITICAL_BUST": "#ef4444" # Red-500
    }
    return mapping.get(risk_level, "#6b7280")


class ConfidenceMapRenderer:
    """
    Renders 2D spatial confidence rasters, GeoJSON vectors, and lead-time forecast curves.
    """

    def __init__(
        self,
        classifier: ForecastBustClassifier | None = None,
        regressor: QuantileErrorRegressor | None = None,
    ):
        self.classifier = classifier
        self.regressor = regressor

    def render_grid_raster(
        self,
        lead_time_days: int = 5,
        bbox: SpatialBoundingBox | None = None,
        resolution_deg: float = 2.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Generate gridded 2D rasters for bust probability, confidence score, and expected error.
        """
        grid_data = generate_spatial_grid_forecast(
            lead_time_days=lead_time_days,
            bbox=bbox,
            resolution_deg=resolution_deg,
            seed=seed,
        )

        lats = grid_data["latitudes"]
        lons = grid_data["longitudes"]
        grid = grid_data["grid"]

        # Flatten grid points for batch inference
        points_flat: list[GridPoint] = []
        feature_rows: list[np.ndarray] = []

        for row in grid:
            for pt_dict in row:
                pt = GridPoint(**pt_dict)
                points_flat.append(pt)

                # Atmospheric features
                feat_dict = {
                    "lead_time_days": float(lead_time_days),
                    "latitude": float(pt.latitude),
                    "longitude": float(pt.longitude),
                    "ensemble_spread_z500": float(pt.ensemble_z500_std),
                    "ensemble_spread_t2m": float(pt.ensemble_t2m_std),
                    "ens_mean_det_diff_z500": float(abs(pt.ensemble_z500_mean - pt.deterministic_z500)),
                    "rossby_wave_activity": float(max(5.0, abs(pt.deterministic_v250) * 1.5)),
                    "blocking_metric": float(pt.blocking_metric),
                    "teleconnection_pna": 0.5,
                    "teleconnection_nao": -0.2,
                    "zonal_wind_shear_250_850": 15.0,
                    "climatological_temp_anomaly": float(pt.deterministic_t2m - 15.0),
                    "day_of_year_sin": 0.2,
                    "day_of_year_cos": 0.9,
                }
                feature_rows.append(extract_atmospheric_features(feat_dict))

        X_mat = np.array(feature_rows, dtype=np.float32)

        # Run models if fitted, otherwise use deterministic physical estimation
        if self.classifier is not None and self.classifier.is_fitted:
            calibrated_probs = self.classifier.predict_proba(X_mat)
            raw_probs = self.classifier.predict_proba_raw(X_mat)
        else:
            # Physics proxy
            spreads = X_mat[:, 3]
            blockings = X_mat[:, 7]
            calibrated_probs = 1.0 / (1.0 + np.exp(-( -3.0 + 0.25 * lead_time_days + 0.04 * spreads + 0.5 * blockings)))
            raw_probs = calibrated_probs

        if self.regressor is not None and self.regressor.is_fitted:
            err_dict = self.regressor.predict_error_quantiles(X_mat)
            expected_z500_err = err_dict["z500_error_m"][0.50]
            expected_t2m_err = err_dict["t2m_error_c"][0.50]
        else:
            expected_z500_err = 12.0 * np.exp(0.18 * lead_time_days) + X_mat[:, 3] * 0.4
            expected_t2m_err = 1.2 * np.exp(0.12 * lead_time_days) + X_mat[:, 4] * 0.3

        n_lat = len(lats)
        n_lon = len(lons)

        prob_matrix = calibrated_probs.reshape((n_lat, n_lon))
        conf_matrix = (1.0 - prob_matrix)
        z500_err_matrix = expected_z500_err.reshape((n_lat, n_lon))
        t2m_err_matrix = expected_t2m_err.reshape((n_lat, n_lon))

        return {
            "lead_time_days": lead_time_days,
            "bbox": grid_data["bbox"],
            "resolution_deg": resolution_deg,
            "shape": [n_lat, n_lon],
            "latitudes": lats,
            "longitudes": lons,
            "calibrated_bust_probability": prob_matrix.round(4).tolist(),
            "raw_bust_probability": raw_probs.reshape((n_lat, n_lon)).round(4).tolist(),
            "confidence_score": conf_matrix.round(4).tolist(),
            "expected_error_z500_m": z500_err_matrix.round(2).tolist(),
            "expected_error_t2m_c": t2m_err_matrix.round(2).tolist(),
        }

    def render_geojson_contours(
        self,
        lead_time_days: int = 5,
        bbox: SpatialBoundingBox | None = None,
        resolution_deg: float = 2.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """
        Generate GeoJSON FeatureCollection of spatial grid polygons colored by bust risk.
        """
        raster = self.render_grid_raster(
            lead_time_days=lead_time_days,
            bbox=bbox,
            resolution_deg=resolution_deg,
            seed=seed,
        )

        lats = raster["latitudes"]
        lons = raster["longitudes"]
        probs = raster["calibrated_bust_probability"]
        errors = raster["expected_error_z500_m"]
        res = raster["resolution_deg"]
        half_res = res / 2.0

        features = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                prob = probs[i][j]
                err = errors[i][j]
                risk = get_risk_level(prob)
                color = get_risk_color(risk)

                # Construct polygon cell boundary
                poly = geojson.Polygon([
                    [
                        [lon - half_res, lat - half_res],
                        [lon + half_res, lat - half_res],
                        [lon + half_res, lat + half_res],
                        [lon - half_res, lat + half_res],
                        [lon - half_res, lat - half_res],
                    ]
                ])

                feat = geojson.Feature(
                    geometry=poly,
                    properties={
                        "latitude": lat,
                        "longitude": lon,
                        "lead_time_days": lead_time_days,
                        "bust_probability": prob,
                        "confidence_score": round(1.0 - prob, 4),
                        "expected_error_z500_m": err,
                        "risk_level": risk,
                        "fill_color": color,
                    },
                )
                features.append(feat)

        return geojson.FeatureCollection(features)

    def compute_location_lead_profile(
        self,
        latitude: float,
        longitude: float,
        init_time: datetime | None = None,
    ) -> LeadTimeForecast:
        """Compute full Day 1 to 10 forecast risk trajectory for a specific location."""
        if init_time is None:
            init_time = datetime.now(timezone.utc)

        days = list(range(1, settings.lead_time_days_max + 1))
        probs: list[float] = []
        confs: list[float] = []
        errors: list[float] = []
        risks: list[str] = []

        for d in days:
            pt = generate_synthetic_grid_point(latitude, longitude, lead_time_days=d, seed=int(latitude * 10 + d))
            feat_dict = {
                "lead_time_days": float(d),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "ensemble_spread_z500": float(pt.ensemble_z500_std),
                "ensemble_spread_t2m": float(pt.ensemble_t2m_std),
                "ens_mean_det_diff_z500": float(abs(pt.ensemble_z500_mean - pt.deterministic_z500)),
                "rossby_wave_activity": float(max(5.0, abs(pt.deterministic_v250) * 1.5)),
                "blocking_metric": float(pt.blocking_metric),
                "teleconnection_pna": 0.5,
                "teleconnection_nao": -0.2,
                "zonal_wind_shear_250_850": 15.0,
                "climatological_temp_anomaly": float(pt.deterministic_t2m - 15.0),
                "day_of_year_sin": 0.2,
                "day_of_year_cos": 0.9,
            }
            x_vec = extract_atmospheric_features(feat_dict).reshape(1, -1)

            if self.classifier and self.classifier.is_fitted:
                p = float(self.classifier.predict_proba(x_vec)[0])
            else:
                p = float(1.0 / (1.0 + np.exp(-(-3.5 + 0.35 * d + 0.03 * pt.ensemble_z500_std + 0.4 * pt.blocking_metric))))

            if self.regressor and self.regressor.is_fitted:
                e = float(self.regressor.predict_error_quantiles(x_vec)["z500_error_m"][0.50][0])
            else:
                e = float(10.0 * np.exp(0.18 * d) + pt.ensemble_z500_std * 0.4)

            probs.append(round(p, 4))
            confs.append(round(1.0 - p, 4))
            errors.append(round(e, 2))
            risks.append(get_risk_level(p))

        return LeadTimeForecast(
            latitude=latitude,
            longitude=longitude,
            init_time=init_time,
            days=days,
            bust_probabilities=probs,
            confidence_scores=confs,
            expected_errors_z500=errors,
            risk_levels=risks,
        )
