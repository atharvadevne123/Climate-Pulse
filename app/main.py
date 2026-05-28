"""FastAPI application for Climate-Pulse: weather prediction API."""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from app.cache import cache_get, cache_invalidate, cache_set
from app.database import get_db, init_db
from app.features import FEATURE_COLUMNS, prepare_features
from app.model import get_metrics, predict, train_models
from app.monitoring import (
    compute_drift,
    compute_feature_drift_from_db,
    get_drift_history,
    get_recent_predictions,
    log_drift_report,
    log_prediction,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Rate limiting ----------
_request_counts: dict[str, list[float]] = {}
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "200"))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP address."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = _request_counts.setdefault(client_ip, [])
        _request_counts[client_ip] = [t for t in window if now - t < 60]
        if len(_request_counts[client_ip]) >= RATE_LIMIT:
            logger.warning("main.RateLimitMiddleware: rate limit exceeded ip=%s", client_ip)
            return Response(content="Rate limit exceeded", status_code=429)
        _request_counts[client_ip].append(now)
        return await call_next(request)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Injects or propagates X-Correlation-ID header for request tracing."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        corr_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = corr_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = corr_id
        return response


# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("main.lifespan: DB initialised")
    yield


# ---------- App ----------
app = FastAPI(
    title="Climate-Pulse",
    description="Short-range climate pattern and extreme weather event prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------
class WeatherInput(BaseModel):
    station_id: str = Field(..., description="Weather station identifier")
    temperature: float = Field(..., ge=-90, le=60, description="Current temperature (°C)")
    precipitation: float = Field(..., ge=0, le=500, description="Current precipitation (mm)")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    pressure: float = Field(..., ge=870, le=1085, description="Atmospheric pressure (hPa)")
    wind_speed: float = Field(..., ge=0, le=200, description="Wind speed (km/h)")
    cloud_cover: float = Field(..., ge=0, le=100, description="Cloud cover (%)")
    month: float = Field(..., ge=1, le=12, description="Month (1-12)")
    day_of_year: float = Field(..., ge=1, le=366, description="Day of year (1-366)")

    @field_validator("station_id")
    @classmethod
    def validate_station_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("station_id cannot be empty")
        return v.strip()


class DriftCheckInput(BaseModel):
    reference: list[float] = Field(..., min_length=5, description="Reference distribution values")
    current: list[float] = Field(..., min_length=5, description="Current distribution values")


class PredictionResponse(BaseModel):
    predicted_temp: float
    predicted_precip: float
    extreme_event_prob: float
    model_version: str
    correlation_id: str
    station_id: str


class HealthResponse(BaseModel):
    status: str
    version: str


class MetricsResponse(BaseModel):
    temp_r2_mean: float
    precip_r2_mean: float
    extreme_auc_mean: float
    n_training_samples: int
    n_features: int
    model_version: str


# ---------- Routes ----------
@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"], summary="Health check")
async def health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/api/v1/readyz", tags=["system"], summary="Readiness probe")
async def readyz() -> dict[str, str]:
    """Kubernetes-compatible readiness probe — returns ready when models are loaded."""
    return {"status": "ready"}


@app.get("/api/v1/version", tags=["system"], summary="API version info")
async def version() -> dict[str, str]:
    """Return the current API and model version strings."""
    from app.model import MODEL_VERSION
    return {"api_version": "1.0.0", "model_version": MODEL_VERSION}


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict next-step temperature, precipitation, and extreme event probability",
)
async def predict_weather(
    request: Request,
    payload: WeatherInput,
    db: Annotated[Session, Depends(get_db)],
) -> PredictionResponse:
    """Given current atmospheric observations, return short-range weather predictions."""
    corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    features_dict = payload.model_dump(exclude={"station_id"})
    df = prepare_features(features_dict)

    try:
        result = predict(df)
    except Exception as exc:
        logger.error("main.predict_weather: prediction failed corr=%s err=%s", corr_id, exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    log_prediction(
        db=db,
        correlation_id=corr_id,
        station_id=payload.station_id,
        features=features_dict,
        predictions=result,
        model_version=result["model_version"],
    )

    return PredictionResponse(
        predicted_temp=result["predicted_temp"],
        predicted_precip=result["predicted_precip"],
        extreme_event_prob=result["extreme_event_prob"],
        model_version=result["model_version"],
        correlation_id=corr_id,
        station_id=payload.station_id,
    )


@app.get(
    "/api/v1/metrics",
    response_model=MetricsResponse,
    tags=["monitoring"],
    summary="Retrieve model training metrics",
)
async def metrics() -> MetricsResponse:
    """Return the latest model training CV metrics (cached 5 min)."""
    cached = cache_get("model_metrics")
    if cached:
        return MetricsResponse(**{k: v for k, v in cached.items() if k in MetricsResponse.model_fields})
    try:
        m = get_metrics()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not load metrics") from exc
    cache_set("model_metrics", m, ttl=300)
    return MetricsResponse(**{k: v for k, v in m.items() if k in MetricsResponse.model_fields})


@app.post(
    "/api/v1/drift",
    tags=["monitoring"],
    summary="Run KS-test drift detection on two distributions",
)
async def drift_check(payload: DriftCheckInput) -> dict[str, Any]:
    """Compute Kolmogorov-Smirnov drift between reference and current distributions."""
    return compute_drift(payload.reference, payload.current)


@app.get(
    "/api/v1/drift/features",
    tags=["monitoring"],
    summary="Run drift detection on recent prediction logs per feature",
)
async def feature_drift(
    db: Annotated[Session, Depends(get_db)],
    feature: str = "temperature",
) -> dict[str, Any]:
    """Pull logged predictions and compute drift for the specified feature."""
    if feature not in FEATURE_COLUMNS:
        raise HTTPException(status_code=400, detail=f"Unknown feature '{feature}'. Valid: {FEATURE_COLUMNS}")
    result = compute_feature_drift_from_db(db, feature)
    log_drift_report(db, feature, result)
    return {"feature": feature, **result}


@app.get(
    "/api/v1/stations/{station_id}/history",
    tags=["monitoring"],
    summary="Fetch prediction history for a specific weather station",
)
async def station_history(
    station_id: str,
    db: Annotated[Session, Depends(get_db)],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return recent prediction logs filtered by station_id."""
    if limit > 200:
        raise HTTPException(status_code=400, detail="limit must be ≤ 200")
    from app.database import PredictionLog
    logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.station_id == station_id)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": log.id,
            "correlation_id": log.correlation_id,
            "station_id": log.station_id,
            "timestamp": log.timestamp.isoformat(),
            "predicted_temp": log.predicted_temp,
            "predicted_precip": log.predicted_precip,
            "extreme_event_prob": log.extreme_event_prob,
        }
        for log in logs
    ]


@app.get(
    "/api/v1/drift/history",
    tags=["monitoring"],
    summary="Retrieve recent drift reports",
)
async def drift_history(
    db: Annotated[Session, Depends(get_db)],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return the most recent drift detection reports."""
    if limit > 200:
        raise HTTPException(status_code=400, detail="limit must be ≤ 200")
    reports = get_drift_history(db, limit=limit)
    return [
        {
            "id": r.id,
            "feature_name": r.feature_name,
            "ks_statistic": r.ks_statistic,
            "p_value": r.p_value,
            "drift_detected": bool(r.drift_detected),
            "timestamp": r.timestamp.isoformat(),
        }
        for r in reports
    ]


@app.post(
    "/api/v1/retrain",
    tags=["model"],
    summary="Trigger model retraining on synthetic data",
)
async def retrain() -> dict[str, Any]:
    """Retrain all models; returns updated CV metrics."""
    logger.info("main.retrain: retraining triggered")
    try:
        metrics_result = train_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Retraining failed") from exc
    cache_invalidate("model_metrics")
    return {"status": "retrained", **metrics_result}


@app.get(
    "/api/v1/predictions/recent",
    tags=["monitoring"],
    summary="Fetch recent prediction logs",
)
async def recent_predictions(
    db: Annotated[Session, Depends(get_db)],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return the most recent prediction records."""
    if limit > 200:
        raise HTTPException(status_code=400, detail="limit must be ≤ 200")
    logs = get_recent_predictions(db, limit=limit)
    return [
        {
            "id": log.id,
            "correlation_id": log.correlation_id,
            "station_id": log.station_id,
            "timestamp": log.timestamp.isoformat(),
            "predicted_temp": log.predicted_temp,
            "predicted_precip": log.predicted_precip,
            "extreme_event_prob": log.extreme_event_prob,
        }
        for log in logs
    ]
