"""Configuration management for BustWatch."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and ML pipeline settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application metadata
    app_name: str = "BustWatch"
    app_env: Literal["development", "staging", "production", "test"] = "development"
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Base Paths
    root_dir: Path = Path(__file__).resolve().parent.parent.parent
    model_dir: Path = Path("artifacts/models")
    evals_dir: Path = Path("evals")
    data_cache_dir: Path = Path("artifacts/cache")

    # Ingestion & Data Sources
    use_synthetic_fallback: bool = True
    noaa_s3_bucket: str = "noaa-gfs-bdp-pds"
    open_meteo_base_url: str = "https://ensemble-api.open-meteo.com/v1/ensemble"
    request_timeout_seconds: int = 15

    # Meteorology & Forecasting Constants
    lead_time_days_min: int = 1
    lead_time_days_max: int = 10
    bust_threshold_z500_m: float = 60.0  # Geopotential height 500hPa error threshold (m)
    bust_threshold_t2m_c: float = 3.5    # 2m temperature error threshold (°C)
    bust_percentile_threshold: float = 90.0  # 90th percentile error

    # ML & Modeling
    random_seed: int = 42
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"
    n_ensemble_members: int = 30
    grid_resolution_deg: float = 1.0


settings = Settings()
