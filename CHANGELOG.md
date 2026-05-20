# Changelog

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
