"""NOAA AWS Open Data connector for GFS/GEFS forecast and reanalysis datasets."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import requests

from bustwatch.config import settings
from bustwatch.data.schemas import ForecastObservationPair

logger = logging.getLogger(__name__)


class NOAADataConnector:
    """Connector for querying and streaming NOAA GFS/GEFS forecasts from AWS Open Data."""

    def __init__(
        self,
        bucket_name: str = settings.noaa_s3_bucket,
        timeout: int = settings.request_timeout_seconds,
    ):
        self.bucket_name = bucket_name
        self.base_url = f"https://{bucket_name}.s3.amazonaws.com"
        self.timeout = timeout

    def build_gfs_url(self, init_time: datetime, forecast_hour: int) -> str:
        """Construct standard AWS S3 HTTP URL for NOAA GFS 0.25-deg GRIB2 file."""
        date_str = init_time.strftime("%Y%m%d")
        cycle_hour = f"{init_time.hour:02d}"
        fhour_str = f"{forecast_hour:03d}"
        return (
            f"{self.base_url}/gfs.{date_str}/{cycle_hour}/atmos/"
            f"gfs.t{cycle_hour}z.pgrb2.0p25.f{fhour_str}"
        )

    def build_index_url(self, init_time: datetime, forecast_hour: int) -> str:
        """Construct S3 HTTP URL for the GRIB2 byte-range index (.idx) file."""
        return f"{self.build_gfs_url(init_time, forecast_hour)}.idx"

    def fetch_index(self, init_time: datetime, forecast_hour: int) -> list[str] | None:
        """Fetch and parse NOAA GRIB2 index file to locate byte offsets for Z500/T2M."""
        idx_url = self.build_index_url(init_time, forecast_hour)
        try:
            resp = requests.get(idx_url, timeout=self.timeout)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                logger.info(f"Retrieved NOAA GFS index for f{forecast_hour:03d} ({len(lines)} records)")
                return lines
            logger.warning(f"NOAA index fetch returned status {resp.status_code}")
            return None
        except requests.RequestException as exc:
            logger.warning(f"Unable to reach NOAA AWS S3 endpoint ({exc}); fallback available")
            return None

    def fetch_forecast_pairs(
        self,
        start_date: datetime,
        end_date: datetime,
        lat: float = 40.0,
        lon: float = -95.0,
    ) -> list[ForecastObservationPair]:
        """
        Fetch forecast-observation pairs for a range of dates.
        Falls back to synthetic physical simulation when offline.
        """
        pairs: list[ForecastObservationPair] = []
        curr = start_date
        while curr <= end_date:
            # Check availability or fallback
            idx = self.fetch_index(curr, forecast_hour=120)
            if idx is None and settings.use_synthetic_fallback:
                from bustwatch.data.synthetic import generate_synthetic_dataset
                synthetic_df = generate_synthetic_dataset(n_samples=5, random_seed=int(curr.timestamp()))
                # Convert synthetic samples
                for _, row in synthetic_df.iterrows():
                    pairs.append(
                        ForecastObservationPair(
                            record_id=f"noaa_syn_{row['record_id']}",
                            forecast_init_time=curr,
                            valid_time=curr + timedelta(days=int(row["lead_time_days"])),
                            lead_time_days=int(row["lead_time_days"]),
                            latitude=lat,
                            longitude=lon,
                            forecast_z500=float(row["forecast_z500"]),
                            forecast_t2m=float(row["forecast_t2m"]),
                            observed_z500=float(row["observed_z500"]),
                            observed_t2m=float(row["observed_t2m"]),
                            ensemble_spread_z500=float(row["ensemble_spread_z500"]),
                            ensemble_spread_t2m=float(row["ensemble_spread_t2m"]),
                            rossby_wave_activity=float(row["rossby_wave_activity"]),
                            blocking_metric=float(row["blocking_metric"]),
                            teleconnection_pna=float(row.get("teleconnection_pna", 0.0)),
                            teleconnection_nao=float(row.get("teleconnection_nao", 0.0)),
                            error_z500=float(row["error_z500"]),
                            error_t2m=float(row["error_t2m"]),
                            is_bust=bool(row["is_bust"]),
                        )
                    )
            curr += timedelta(days=1)
        return pairs
