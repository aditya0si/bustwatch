"""Main FastAPI Application for BustWatch."""

from __future__ import annotations
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bustwatch.config import settings
from bustwatch.models.bust_classifier import ForecastBustClassifier
from bustwatch.models.error_regressor import QuantileErrorRegressor
from bustwatch.renderer.confidence_map import ConfidenceMapRenderer
from bustwatch.api.routes import forecast, maps, evals

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan management."""
    logger.info("Initializing BustWatch ML Models and Rendering Engine...")

    clf_path = settings.model_dir / "bust_classifier.joblib"
    reg_path = settings.model_dir / "error_regressor.joblib"

    # Load or initialize classifier
    if clf_path.exists():
        logger.info(f"Loading trained ForecastBustClassifier from {clf_path}")
        classifier = ForecastBustClassifier.load(clf_path)
    else:
        logger.warning(f"No saved classifier found at {clf_path}; training lightweight fallback model...")
        from scripts.train_models import train_and_save_models
        train_and_save_models(n_samples=500, output_dir=settings.model_dir)
        classifier = ForecastBustClassifier.load(clf_path)

    # Load or initialize regressor
    if reg_path.exists():
        logger.info(f"Loading trained QuantileErrorRegressor from {reg_path}")
        regressor = QuantileErrorRegressor.load(reg_path)
    else:
        logger.warning(f"No saved regressor found at {reg_path}; initializing fallback...")
        regressor = QuantileErrorRegressor()

    renderer = ConfidenceMapRenderer(classifier=classifier, regressor=regressor)

    app.state.classifier = classifier
    app.state.regressor = regressor
    app.state.renderer = renderer
    logger.info("BustWatch API initialized and ready for real-time inference.")

    yield

    logger.info("Shutting down BustWatch API.")


app = FastAPI(
    title="BustWatch API",
    description="Medium-Range NWP Forecast Bust Detection & Confidence Mapping Engine",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-routers
app.include_router(forecast.router, prefix=settings.api_v1_prefix)
app.include_router(maps.router, prefix=settings.api_v1_prefix)
app.include_router(evals.router, prefix=settings.api_v1_prefix)


@app.get("/health", tags=["System Health"])
async def health_check():
    """System health and model readiness probe."""
    uptime_sec = time.time() - START_TIME
    is_clf_ready = hasattr(app.state, "classifier") and app.state.classifier is not None
    is_reg_ready = hasattr(app.state, "regressor") and app.state.regressor is not None

    return {
        "status": "healthy" if (is_clf_ready and is_reg_ready) else "degraded",
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "uptime_seconds": round(uptime_sec, 2),
        "models": {
            "bust_classifier_loaded": is_clf_ready,
            "error_regressor_loaded": is_reg_ready,
            "calibration_method": settings.calibration_method,
        },
        "version": "0.1.0",
    }


@app.get("/", tags=["System Health"])
async def root():
    """Root metadata endpoint."""
    return {
        "message": "Welcome to BustWatch Medium-Range NWP Forecast Bust Detection API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict_bust": f"{settings.api_v1_prefix}/forecast/bust-risk",
            "lead_profile": f"{settings.api_v1_prefix}/forecast/lead-profile",
            "confidence_grid": f"{settings.api_v1_prefix}/maps/confidence-grid",
            "geojson_map": f"{settings.api_v1_prefix}/maps/geojson",
            "eval_metrics": f"{settings.api_v1_prefix}/evals/metrics",
            "historical_busts": f"{settings.api_v1_prefix}/evals/historical-busts",
        },
    }
