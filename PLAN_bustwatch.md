# PLAN_bustwatch.md — Medium-Range NWP Forecast Bust Detection & Day 1–10 Confidence Mapping System

---

## Section A — Goal & Acceptance Criteria

### Technical Restatement
BustWatch (`bustwatch`) is a machine learning system and operational platform designed to detect Numerical Weather Prediction (NWP) forecast busts (extreme forecast error events exceeding regional/seasonal 90th percentile error thresholds) across Day 1 to Day 10 (24h to 240h) lead times. The system combines NWP model states (GFS/GEFS/ECMWF), ensemble spread, atmospheric circulation indices (Rossby wave activity, blocking patterns, teleconnections), and physical geographical covariates. It trains calibrated Gradient Boosted Decision Trees (LightGBM/HistGradientBoosting) with Platt/Isotonic probability calibration, outputs gridded spatial confidence and bust risk maps, serves real-time predictions via a high-throughput FastAPI REST API, provides an interactive React+Vite web interface with Leaflet/Canvas confidence heatmaps and reliability visualizers, and includes an evaluation framework calculating Brier scores, Brier Skill Scores (BSS), Expected Calibration Error (ECE), and reliability diagrams.

### Observable "Done" Definition
1. **Data Pipeline**: End-to-end data acquisition module connecting to NOAA AWS Open Data / Open-Meteo Historical Forecast APIs with a high-fidelity synthetic/benchmark physical generator for reproducible, fast local runs.
2. **ML Engine & Calibration**: Gradient boosting models for bust classification ($P(\text{Bust} = 1)$) and quantile error regression across lead times $t \in [1, 10]$ days, equipped with probability calibration (Isotonic/Platt) achieving lower Brier score than uncalibrated baselines.
3. **Confidence Map Renderer**: Spatio-temporal grid generator producing 2D confidence rasters, GeoJSON vectors, and uncertainty contour maps for Days 1–10.
4. **Backend REST API**: FastAPI application with `/health`, `/api/v1/forecast/bust-risk`, `/api/v1/maps/confidence-grid`, `/api/v1/evals/metrics`, and data sync endpoints passing all validation schemas.
5. **Frontend Application**: React + Vite + Tailwind CSS + Lucide Icons + Leaflet/Canvas dashboard for interactive Day 1–10 lead-time scrubbing, station inspection, and calibration diagram exploration.
6. **Evaluation Framework (`evals/`)**: Standalone, runnable script `evals/evaluate_models.py` generating Brier Score, Brier Skill Score, ECE, reliability curves (JSON & figures), and populating the README benchmark table.
7. **Quality & CI**: Full `pytest` test suite covering core data logic, ML pipelines, map rendering, and API endpoints with 100% passing tests; GitHub Actions CI workflow configured for linting and testing.
8. **Git Repository**: Initialized git repo with initial commit `feat: initial implementation`.

### Out of Scope
- Direct distributed training on multi-terabyte raw GRIB2 archives (a streaming sample pipeline and synthetic benchmark are provided).
- Operational deployment to Kubernetes/Cloud clusters (Docker and local FastAPI/Vite dev configurations are provided).

---

## Section B — Tech Stack & Constraints

### Languages & Frameworks
- **Python 3.10+**: Core backend, ML, data engineering, evaluation framework.
- **FastAPI + Uvicorn + Pydantic v2**: High-performance asynchronous REST API.
- **TypeScript + React 18 + Vite + Tailwind CSS + Lucide React + Leaflet/Recharts**: Modern interactive frontend.
- **Scientific & ML Stack**: `numpy`, `scipy`, `pandas`, `scikit-learn`, `lightgbm` (with fallback to `HistGradientBoostingClassifier`), `joblib`.
- **Geospatial & Evaluation Tools**: `geojson`, `matplotlib` (for headless reliability diagram generation).
- **Testing & Tooling**: `pytest`, `pytest-asyncio`, `httpx`, `ruff` / `flake8`.

### Architectural Decisions & Alternatives Considered
1. **Gradient Boosting vs. Deep Neural Networks for Bust Detection**:
   - *Decision*: Gradient Boosted Trees (LightGBM/HistGradientBoosting).
   - *Rationale*: Tabular atmospheric regime features, lead times, and ensemble spread have strong non-linear threshold effects. GBDTs train quickly, produce robust feature importances, and avoid catastrophic overfitting on sparse extreme bust events compared to heavy 3D ConvNets.
2. **Probability Calibration (Isotonic vs. Platt Scaling)**:
   - *Decision*: Hybrid calibration support (Isotonic Regression for non-parametric calibration when sample size is large; Sigmoid/Platt for smaller sample sizes).
   - *Rationale*: Raw tree outputs suffer from poor probability calibration near extreme tails (0 and 1).
3. **Data Pipeline Architecture**:
   - *Decision*: Dual-mode data connector (live Open-Meteo & NOAA GFS connector + deterministic physical-statistical synthetic generator).
   - *Rationale*: Ensures zero-dependency, rapid testing and offline local execution while providing real API ingestion code for production scale-up.

### Stack Touchpoints
- **Created**: `src/bustwatch/` (core data, models, rendering, API), `frontend/` (React+Vite web app), `evals/` (eval scripts and benchmarks), `tests/` (unit & integration tests), `scripts/` (CLI tools), `.github/workflows/ci.yml`, `README.md`, `LICENSE`, `pyproject.toml`, `requirements.txt`, `.env.example`.
- **Untouched**: Root repo metadata.

---

## Section C — Blocking Questions (0–3) & Assumptions

### Blocking Questions
`None (0)` — Requirements and technical specification are fully defined.

### Falsifiable Assumptions
- `[ASSUMPTION 1: Data Domain]` A "forecast bust" is defined operationally when the forecast error (e.g. 500hPa geopotential height error $\ge 60\text{ m}$ or 2m temperature error $\ge 3.5^\circ\text{C}$ or regional 90th percentile) occurs at lead times 1 to 10 days.
- `[ASSUMPTION 2: Failure Domain]` If remote meteorological data sources (Open-Meteo / NOAA S3) are unreachable or rate-limited, the system falls back to the deterministic synthetic physical generator without failing tests or evaluation runs.
- `[ASSUMPTION 3: Boundary Domain]` The API will serve both GeoJSON vector features and dense grid arrays (JSON) so frontend consumers can render fast Canvas heatmaps or interactive Leaflet layers.
- `[ASSUMPTION 4: State & Concurrency]` Pretrained/cached model artifacts will be loaded in memory upon FastAPI startup for sub-50ms inference per query.
- `[ASSUMPTION 5: Environment]` Python 3.10+ and standard Node/npm environments are available on Windows OS.
- `[ASSUMPTION 6: Testing]` All core math, calibration, rendering algorithms, and FastAPI endpoints will be tested deterministically with pytest.

---

## Section D — Session Modularization

### Session 1: Project Architecture, Repository Layout & Dependencies
- **Objective**: Establish repo scaffolding, `pyproject.toml`, `requirements.txt`, `.env.example`, `LICENSE`, and GitHub Actions CI workflow.
- **Scope**: Root directory metadata and build files.
- **Output**: Clean packaging config and runnable pytest environment.
- **Connects To**: Session 2.
- **Failure Surface**: Dependency version mismatches.

### Session 2: Data Pipeline & Atmospheric Dynamics Engine
- **Objective**: Build historical forecast error ingestion (Open-Meteo & NOAA connectors) and synthetic meteorological error simulation (Rossby wave dispersion, ensemble spread, regime shifts).
- **Scope**: `src/bustwatch/data/` (connectors, feature engineering, schemas, synthetic generator).
- **Output**: Dataset generator producing calibrated train/val/test splits with lead times 1–10 days.
- **Connects To**: Session 3.
- **Failure Surface**: Data leakage across lead times or train/test splits.

### Session 3: Machine Learning Engine, Probability Calibration & Confidence Mapping
- **Objective**: Implement GBDT classifiers/regressors, Platt/Isotonic calibration, and the Day 1–10 Confidence Map Renderer.
- **Scope**: `src/bustwatch/models/`, `src/bustwatch/renderer/`, `src/bustwatch/calibration/`.
- **Output**: Model training pipeline, serialized model artifacts, and spatial grid rendering engine.
- **Connects To**: Session 4 & Session 5.
- **Failure Surface**: Poor calibration slope or out-of-bounds probability outputs.

### Session 4: Evaluation Benchmark Suite & Visualization
- **Objective**: Create `evals/evaluate_models.py` to calculate Brier score, Brier Skill Score (BSS), ECE, ROC-AUC, and generate reliability diagrams.
- **Scope**: `evals/` scripts and automated metrics reporting.
- **Output**: Runnable evaluation benchmark script producing metrics table and plot assets.
- **Connects To**: Session 5 & Documentation.
- **Failure Surface**: Missing metric edge cases (e.g. division by zero in skill score).

### Session 5: FastAPI Backend & React+Vite Frontend Dashboard
- **Objective**: Implement REST API endpoints (`/health`, `/api/v1/forecast/bust-risk`, `/api/v1/maps/confidence-grid`, `/api/v1/evals/metrics`) and responsive React dashboard.
- **Scope**: `src/bustwatch/api/` and `frontend/`.
- **Output**: Fully functional backend and modern interactive frontend with Day 1–10 slider, confidence heatmaps, and reliability visualizer.
- **Connects To**: Session 6.
- **Failure Surface**: CORS misconfigurations or slow raster serialization.

### Session 6: Test Suite, Documentation & Git Commit
- **Objective**: Complete comprehensive pytest suite, write detailed `README.md` (with Mermaid architecture, math formulation, metrics table, setup guide), verify 100% tests pass, and initialize Git repo with initial commit.
- **Scope**: `tests/`, `README.md`, git init.
- **Output**: All tests pass, GitHub-ready portfolio repository.
- **Connects To**: Final summary delivery.
- **Failure Surface**: Test failures or uncommitted files.

---

## Section E — Progress Checklist

- [ ] Session 1: Project Scaffolding & Packaging
  - [ ] `pyproject.toml`, `requirements.txt`, `.env.example`, `LICENSE`
  - [ ] `.github/workflows/ci.yml`
- [ ] Session 2: Data Pipeline & Synthetic Meteorology Engine
  - [ ] `src/bustwatch/data/schemas.py` & `features.py`
  - [ ] `src/bustwatch/data/noaa_connector.py` & `openmeteo_connector.py`
  - [ ] `src/bustwatch/data/synthetic.py`
  - [ ] Unit tests for data generation and feature extraction
- [ ] Session 3: ML Models, Calibration & Map Renderer
  - [ ] `src/bustwatch/models/bust_classifier.py` & `error_regressor.py`
  - [ ] `src/bustwatch/calibration/calibrator.py`
  - [ ] `src/bustwatch/renderer/confidence_map.py`
  - [ ] Model training script & artifact generation
- [ ] Session 4: Evaluation Benchmark Suite
  - [ ] `evals/evaluate_models.py` (Brier score, BSS, ECE, Reliability diagrams)
  - [ ] Metrics generation and JSON/table export
- [ ] Session 5: Backend API & Frontend Dashboard
  - [ ] `src/bustwatch/api/main.py` & routers
  - [ ] `frontend/` React + Vite + Tailwind + Leaflet dashboard
  - [ ] API integration tests
- [ ] Session 6: Testing, Documentation & Git Finalization
  - [ ] Complete `tests/` suite (unit, integration, end-to-end)
  - [ ] Comprehensive `README.md` with Mermaid diagram, metrics table, and setup
  - [ ] Run `pytest` and verify all tests pass
  - [ ] Git init & initial commit (`feat: initial implementation`)
