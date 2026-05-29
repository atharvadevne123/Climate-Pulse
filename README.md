# Climate-Pulse

[![CI](https://github.com/atharvadevne123/Climate-Pulse/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/Climate-Pulse/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-75%25%2B-brightgreen.svg)](https://github.com/atharvadevne123/Climate-Pulse/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Short-range climate pattern and extreme weather event prediction API using XGBoost-LightGBM ensemble with atmospheric feature engineering, KS-drift monitoring, and Airflow retraining pipelines.**

---

## Overview

Climate-Pulse is a production-ready ML system that predicts next-step temperature, precipitation amounts, and extreme weather event probability (heatwaves, frost events, heavy precipitation) from atmospheric observations.

### Key Capabilities

- **16 REST endpoints** under `/api/v1/` for prediction, drift monitoring, retraining, and observability
- **XGBoost + LightGBM + RandomForest ensemble** for temperature, precipitation, and extreme-event classification
- **6-stage feature pipeline**: lag features (3 steps), rolling stats (3/7/14 windows), atmospheric ratios (humidity-pressure ratio, wind chill), seasonal sine/cosine encoding, dew-point calculation, heat index (Rothfusz)
- **5-fold cross-validation** with R² for regression targets and AUC-ROC for extreme event detection
- **KS-test drift detection** per feature column with DB-backed reports
- **Airflow weekly retraining DAG** with R²/AUC quality gates
- **Docker + PostgreSQL** production setup
- **In-process model bundle cache** to avoid repeated disk reads on every prediction
- **200+ pytest tests** across 14 test modules

---

## Architecture

![Architecture](screenshots/architecture.png)

```
HTTP Request
     │
     ▼
RateLimitMiddleware ──► CorrelationIDMiddleware
     │
     ▼
FastAPI Router /api/v1/
     │
     ├─► predict ──► FeaturePipeline (6 stages) ──► EnsembleModel (XGB+LGBM+RF)
     │                                                      │
     │                                                      ▼
     │                                               PredictionLog (SQLite/PostgreSQL)
     │
     ├─► drift ──► KS-test ──► DriftReport
     │
     └─► telemetry/cache/monitoring endpoints
```

---

## Quick Start

### Local (SQLite)

```bash
git clone https://github.com/atharvadevne123/Climate-Pulse
cd Climate-Pulse
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

### Docker (PostgreSQL)

```bash
docker-compose up --build
```

API available at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`

---

## API Reference

### Prediction

#### `POST /api/v1/predict`

Predict next-step temperature, precipitation, and extreme event probability.

**Request:**
```json
{
  "station_id": "STATION_001",
  "temperature": 22.5,
  "precipitation": 3.2,
  "humidity": 65.0,
  "pressure": 1012.5,
  "wind_speed": 18.0,
  "cloud_cover": 40.0,
  "month": 6.0,
  "day_of_year": 160.0
}
```

**Response:**
```json
{
  "predicted_temp": 23.1,
  "predicted_precip": 2.8,
  "extreme_event_prob": 0.0312,
  "model_version": "1.0.0",
  "correlation_id": "abc-123",
  "station_id": "STATION_001"
}
```

### Model

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/metrics` | CV metrics (R², AUC-ROC, sample count) |
| `GET /api/v1/model/info` | Pipeline stages, input features, is-trained flag |
| `GET /api/v1/model/freshness` | Age of persisted model files |
| `POST /api/v1/retrain` | Trigger retraining; returns updated metrics |
| `GET /api/v1/version` | API and model version strings |

### Monitoring

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/drift` | KS-test between two distributions |
| `GET /api/v1/drift/features?feature=temperature` | Per-feature drift from DB logs |
| `GET /api/v1/drift/history?limit=20` | Recent drift reports |
| `GET /api/v1/drift/summary` | Drift event counts per feature |
| `GET /api/v1/predictions/recent?limit=20` | Recent prediction logs |
| `DELETE /api/v1/predictions/purge?keep_latest=10000` | Purge old logs |
| `GET /api/v1/stations/{station_id}/history` | Station prediction history |
| `GET /api/v1/stations/{station_id}/stats` | Aggregate station statistics |

### System

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/health` | Health check |
| `GET /api/v1/readyz` | Kubernetes readiness probe |
| `GET /api/v1/telemetry` | Request counters and latency percentiles |
| `GET /api/v1/cache/stats` | In-memory cache hit rate and size |

---

## Feature Engineering Pipeline

| Stage | Transformer | Output Features |
|-------|------------|-----------------|
| 1 | `LagFeatureTransformer` | temp_lag_1/2/3, precip_lag_1/2/3 |
| 2 | `RollingStatsTransformer` | temp_roll_mean/std (3,7,14), precip_roll_sum (3,7,14) |
| 3 | `AtmosphericRatioTransformer` | humidity_pressure_ratio, wind_chill |
| 4 | `SeasonalEncodingTransformer` | month_sin, month_cos, doy_sin, doy_cos |
| 5 | `DewiPointTransformer` | dew_point |
| 6 | `HeatIndexTransformer` | heat_index (Rothfusz equation) |
| 7 | `StandardScaler` | scaled features |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=term-missing --cov-fail-under=75

# Run specific test module
pytest tests/test_api.py -v
pytest tests/test_model.py -v
pytest tests/test_features.py -v
```

The test suite includes 200+ tests across 14 modules:

| Module | Coverage |
|--------|---------|
| `test_api.py`, `test_api_extended.py` | Endpoint happy paths, edge cases, parametrized |
| `test_model.py`, `test_model_extended.py` | Training, prediction, ensemble, extremes |
| `test_features.py` | Feature pipeline and all 6 transformers |
| `test_monitoring.py` | Drift detection, station stats, purge |
| `test_database.py` | ORM CRUD, helper functions |
| `test_cache.py` | TTL, hit/miss tracking, cache_get_or_set |
| `test_validators.py` | Input validation, cross-field checks |
| `test_telemetry.py` | Counters, histograms, snapshot |
| `test_utils.py` | All utility functions |
| `test_new_endpoints.py` | Readyz, version, purge, freshness |
| `test_retrain.py` | Retrain endpoint flow |

---

## Performance

- **Model bundle caching**: models loaded from disk once and cached in-process — subsequent predictions skip I/O
- **Composite DB indexes**: `(station_id, timestamp)` and `(feature_name, timestamp)` for fast history and drift queries
- **TTL in-memory cache**: metrics endpoint cached for 5 minutes to avoid repeated model file reads
- **Connection pooling**: configurable via `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT`

---

## Development

```bash
make install      # install dependencies
make test         # run pytest with coverage
make lint         # ruff check + format check
make lint-fix     # auto-fix ruff issues
make type-check   # mypy type checking
make run          # start dev server
make benchmark    # run load test script
make retrain      # trigger live API retrain
make db-migrate   # apply alembic migrations
make diagram      # regenerate architecture diagram
make clean        # remove build artifacts and cache
```

---

## Project Structure

```
Climate-Pulse/
├── app/
│   ├── main.py           # FastAPI app, middleware, 16 endpoints
│   ├── model.py          # Ensemble ML training, prediction, bundle cache
│   ├── features.py       # 6-stage feature engineering pipeline
│   ├── monitoring.py     # KS-drift detection, station stats, purge
│   ├── database.py       # SQLAlchemy models, session, helper queries
│   ├── cache.py          # TTL cache with hit/miss tracking
│   ├── telemetry.py      # In-process counters and latency histograms
│   ├── validators.py     # Input and output validation
│   ├── utils.py          # General-purpose utilities
│   ├── config.py         # Settings from environment variables
│   ├── exceptions.py     # Custom exception handlers
│   └── logging_config.py # Structured key=value log formatter
├── alembic/              # Database migrations
├── pipelines/
│   └── retrain_dag.py    # Airflow weekly retraining DAG
├── tests/                # 200+ pytest tests across 14 modules
├── scripts/
│   ├── generate_diagram.py
│   ├── load_test.py
│   └── retrain.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
