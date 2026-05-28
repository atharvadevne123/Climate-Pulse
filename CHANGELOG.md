# Changelog

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
