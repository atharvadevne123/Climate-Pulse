# Changelog

## [1.2.0] - 2026-05-29

### Added
- `HeatIndexTransformer` (Rothfusz regression) added as 6th stage in feature pipeline
- `/api/v1/model/info` — pipeline stages, input features, is-trained status
- `/api/v1/stations/{station_id}/stats` — aggregate prediction statistics per station
- `/api/v1/drift/summary` — drift event counts grouped by feature
- `/api/v1/cache/stats` — in-memory cache hit rate and size
- `/api/v1/predictions/purge` — DELETE endpoint to purge old prediction logs
- `cache_get_or_set()` convenience function with hit/miss tracking
- `cache_stats()` and `cache_hit_rate()` cache observability functions
- `get_station_stats()` aggregate summary in monitoring module
- `get_drift_count_by_feature()` grouping function in monitoring module
- `purge_old_predictions()` maintenance function in monitoring module
- `reset_model_cache()` and `is_model_trained()` in model module
- In-process model bundle cache to avoid repeated joblib.load() on every predict call
- `percentage_change()`, `format_duration()`, `batch_iter()` utility functions
- `validate_station_id_format()` and `validate_prediction_output()` validators
- `get_counter_names()` and `snapshot()` in telemetry module
- `get_oldest_prediction()` and `get_prediction_count_by_station()` DB helpers
- Alembic migration 002: composite indexes on station+timestamp and feature+timestamp
- `benchmark`, `retrain`, `db-migrate` Makefile targets
- 150+ additional tests across 12 test files

### Fixed
- Deprecated `datetime.utcnow` in ORM column defaults replaced with lambda `datetime.now(UTC)`
- Ruff format compliance for all source files

## [1.1.0] - 2026-05-28

### Added
- `/api/v1/readyz` — Kubernetes readiness probe
- `/api/v1/version` — API and model version endpoint
- `/api/v1/drift/history` — retrieve recent drift detection reports
- `/api/v1/stations/{station_id}/history` — per-station prediction history
- Connection pool configuration (DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_TIMEOUT)
- `get_drift_history()` helper in monitoring module
- `_transform_features()` helper in model module
- `STRUCTURED_FIELDS` constant in logging_config module
- `STATION_ID_MAX_LEN` constant and max-length check in validators module
- pytest-cov coverage threshold (75%) added to CI
- mypy type checking job in CI
- bandit security scanning job in CI
- Multi-Python (3.11, 3.12) matrix in CI
- 180+ new tests across 7 new test modules

### Fixed
- `datetime.utcnow()` replaced with timezone-aware `datetime.now(UTC)` in monitoring module
- DB session not rolling back between tests (conftest.py isolation fix)
- `lead_time_days=0` incorrectly using default 30 instead of clamping to minimum 1 (Supply-Pulse)

### Changed
- `build_feature_pipeline()` now cached with `lru_cache(maxsize=1)`
- CI timeout increased to 30 minutes with per-test timeout of 120s
- README updated with testing section, new endpoints, and additional badges

## [1.0.0] - 2026-05-20

### Added
- FastAPI REST API with 7 endpoints under `/api/v1/`
- XGBoost + LightGBM + RandomForest ensemble for temperature, precipitation, and extreme event prediction
- 5-stage sklearn feature engineering pipeline: lag features, rolling stats, atmospheric ratios, seasonal encoding, dew point
- 5-fold cross-validation with R² and AUC-ROC metrics
- KS-test drift detection per feature column
- SQLAlchemy ORM with PredictionLog, DriftReport, WeatherObservation models
- PostgreSQL + docker-compose production setup
- Airflow weekly retraining DAG with validation gates
- Rate limiting middleware (200 req/min per IP)
- Correlation ID middleware for distributed tracing
- 40+ pytest tests across 4 test modules
- GitHub Actions CI with ruff lint and pytest
- Architecture diagram in `screenshots/architecture.png`
