# ML Methodology

## Problem Framing

Climate-Pulse treats short-range weather prediction as three supervised learning tasks:
1. **Temperature regression** — predict next-step temperature (°C)
2. **Precipitation regression** — predict next-step precipitation (mm, clipped ≥ 0)
3. **Extreme event classification** — binary probability of heatwave/frost/heavy rain

## Feature Engineering

| Feature Group | Rationale |
|---------------|-----------|
| Lag-1/2/3 temperature | Short-term autocorrelation in temperature series |
| Rolling mean/std (3,7,14) | Trend smoothing and volatility estimation |
| Humidity-pressure ratio | Convective instability indicator |
| Wind chill index | Perceived temperature correction for wind |
| Sine/cosine month encoding | Captures seasonal periodicity without ordinal bias |
| Dew point (Magnus approximation) | Key predictor for fog and frost formation |

## Model Architecture

All three tasks use a **homogeneous ensemble** of:
- **LightGBM** — fast gradient boosting, strong on tabular data with many features
- **XGBoost** — robust to noisy features; excellent on short time-series features
- **RandomForest** — high-variance reducer; provides diversity in the ensemble

Predictions are averaged (regression) or soft-voted (classification) from the three models.

## Training Procedure

- 2 000 synthetic observations generated with seasonal temperature pattern + noise
- 5-fold cross-validation: KFold for regression, StratifiedKFold for classification
- Metrics: R² (temperature, precipitation), AUC-ROC (extreme events)
- Models persisted with `joblib` for fast loading at inference time

## Monitoring

KS-test on each input feature compares:
- **Reference**: last 500 logged predictions
- **Current**: last 100 logged predictions
- `drift_detected = True` when `p < 0.05`

Retraining is triggered weekly via the Airflow DAG, with a quality gate requiring R² > 0.50 and AUC > 0.70 before models are promoted.
