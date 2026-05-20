# Architecture

## Component Overview

```
Client → FastAPI (Rate Limit + CorrID Middleware)
              ↓
        Feature Pipeline
        (Lag · Rolling · Ratios · Seasonal · DewPoint)
              ↓
        ML Ensemble
        (XGBoost + LightGBM + RandomForest)
         ├── Temperature Regressor
         ├── Precipitation Regressor
         └── Extreme Event Classifier
              ↓
        Prediction Response
              ↓
        Monitoring (KS-Drift, Prediction Logs → DB)
```

## Data Flow

1. **Request** — Client sends atmospheric observations to `/api/v1/predict`
2. **Validation** — Pydantic validates all fields with range constraints
3. **Feature Engineering** — 5-stage pipeline adds lag, rolling, ratio, seasonal, and dew-point features
4. **Inference** — Three separate ensemble models predict temperature, precipitation, extreme-event probability
5. **Logging** — Prediction stored in `prediction_logs` with correlation ID
6. **Response** — Structured JSON with predictions and metadata

## Models

| Model | Target | Algorithm | CV Score |
|-------|--------|-----------|----------|
| Temperature | Next-step °C | XGB+LGBM+RF avg | R² ≈ 0.97 |
| Precipitation | Next-step mm | XGB+LGBM+RF avg | R² ≈ 0.76 |
| Extreme Event | Binary class | XGB+LGBM+RF soft-vote | AUC ≈ 0.99 |

## Drift Detection

KS-test compares reference distribution (last 500 predictions) against recent window (last 100). `p < 0.05` triggers a drift alert logged to `drift_reports`.

## Airflow DAG

Weekly retraining DAG with 4 tasks:
1. `load_data` — pull latest observations
2. `train_models` — fit all 3 ensembles
3. `validate_models` — assert R² > 0.5, AUC > 0.7
4. `log_metrics` — record metrics to logs
