"""Evaluation metrics, reliability curves, and historical bust case studies API routes."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException

from bustwatch.config import settings
from bustwatch.data.schemas import BustRecord

router = APIRouter(prefix="/evals", tags=["Model Verification & Evals"])

HISTORICAL_BUSTS: List[BustRecord] = [
    BustRecord(
        event_id="bust_2021_texas_freeze",
        date=datetime(2021, 2, 13, 0, 0),
        lead_time_days=7,
        location_name="Texas / Southern Plains, USA",
        latitude=31.9686,
        longitude=-99.9018,
        bust_type="BLOCKING_COLLAPSE",
        observed_error_z500=145.0,
        calibrated_bust_prob=0.84,
        description="Arctic polar vortex elongation and Greenland blocking ridge failure caused severe 7-day lead time temperature underestimation of 14°C across ERCOT power grid.",
    ),
    BustRecord(
        event_id="bust_2022_uk_heatwave",
        date=datetime(2022, 7, 15, 0, 0),
        lead_time_days=6,
        location_name="London / Southeast England, UK",
        latitude=51.5074,
        longitude=-0.1278,
        bust_type="ROSSBY_WAVE_BREAKING",
        observed_error_z500=112.0,
        calibrated_bust_prob=0.76,
        description="Cut-off Iberian low-pressure system and subtropical ridge amplification led to historic 40.3°C temperatures poorly resolved at Day 6-8 lead times.",
    ),
    BustRecord(
        event_id="bust_2024_california_ar",
        date=datetime(2024, 2, 4, 0, 0),
        lead_time_days=5,
        location_name="Sierra Nevada / Central California, USA",
        latitude=36.7783,
        longitude=-119.4179,
        bust_type="CONVECTIVE_FEEDBACK",
        observed_error_z500=98.0,
        calibrated_bust_prob=0.69,
        description="Pineapple Express atmospheric river mesoscale frontal wave development diverged from ensemble mean tracks by over 400 km.",
    ),
]


@router.get("/metrics")
async def get_eval_metrics() -> Dict[str, Any]:
    """Retrieve verified evaluation metrics, Brier scores, ECE, and reliability curve coordinates."""
    metrics_path = settings.evals_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)

    # If file does not exist, compute lightweight verification on demand
    from evals.evaluate_models import run_evaluations
    return run_evaluations(n_samples=300)


@router.get("/historical-busts", response_model=List[BustRecord])
async def list_historical_busts():
    """Retrieve catalog of benchmark historical NWP forecast bust case studies."""
    return HISTORICAL_BUSTS
