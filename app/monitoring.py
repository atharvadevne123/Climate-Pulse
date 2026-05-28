"""Drift detection, prediction logging, and model monitoring."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftReport, PredictionLog

logger = logging.getLogger(__name__)


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Kolmogorov-Smirnov test between reference and current distributions.

    Returns drift detected when p-value < 0.05.
    """
    if len(reference) < 5 or len(current) < 5:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "reason": "insufficient_data"}
    stat, p = ks_2samp(reference, current)
    result = {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }
    if result["drift_detected"]:
        logger.warning("monitoring.compute_drift: DRIFT DETECTED ks=%.4f p=%.4f", stat, p)
    return result


def log_prediction(
    db: Session,
    correlation_id: str,
    station_id: str,
    features: dict[str, Any],
    predictions: dict[str, Any],
    model_version: str,
) -> PredictionLog:
    """Persist a prediction record to the database."""
    record = PredictionLog(
        correlation_id=correlation_id,
        timestamp=datetime.now(UTC),
        station_id=station_id,
        features=features,
        predicted_temp=predictions.get("predicted_temp"),
        predicted_precip=predictions.get("predicted_precip"),
        extreme_event_prob=predictions.get("extreme_event_prob"),
        model_version=model_version,
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as exc:
        db.rollback()
        logger.error("monitoring.log_prediction: DB write failed corr=%s err=%s", correlation_id, exc)
        raise
    logger.info(
        "monitoring.log_prediction: id=%d station=%s corr=%s",
        record.id,
        station_id,
        correlation_id,
    )
    return record


def log_drift_report(
    db: Session,
    feature_name: str,
    drift_result: dict[str, Any],
) -> DriftReport:
    """Persist a drift report for a single feature."""
    report = DriftReport(
        timestamp=datetime.now(UTC),
        feature_name=feature_name,
        ks_statistic=drift_result["ks_statistic"],
        p_value=drift_result["p_value"],
        drift_detected=int(drift_result["drift_detected"]),
    )
    try:
        db.add(report)
        db.commit()
        db.refresh(report)
    except Exception as exc:
        db.rollback()
        logger.error("monitoring.log_drift_report: DB write failed feature=%s err=%s", feature_name, exc)
        raise
    logger.info(
        "monitoring.log_drift_report: feature=%s drift=%s ks=%.4f",
        feature_name,
        drift_result["drift_detected"],
        drift_result["ks_statistic"],
    )
    return report


def get_recent_predictions(db: Session, limit: int = 100) -> list[PredictionLog]:
    """Fetch the most recent prediction logs ordered by timestamp desc."""
    return db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()


def get_prediction_by_correlation_id(db: Session, correlation_id: str) -> PredictionLog | None:
    """Return a single prediction log by correlation ID, or None if not found.

    Args:
        db: Active SQLAlchemy session.
        correlation_id: UUID string assigned to the original prediction request.

    Returns:
        Matching PredictionLog instance, or None.
    """
    return (
        db.query(PredictionLog)
        .filter(PredictionLog.correlation_id == correlation_id)
        .first()
    )


def get_drift_history(db: Session, limit: int = 100) -> list[DriftReport]:
    """Fetch the most recent drift reports ordered by timestamp desc.

    Args:
        db: Database session.
        limit: Maximum number of records to return.

    Returns:
        List of DriftReport ORM instances.
    """
    return db.query(DriftReport).order_by(DriftReport.timestamp.desc()).limit(limit).all()


def compute_feature_drift_from_db(
    db: Session,
    feature_name: str,
    reference_window: int = 500,
    current_window: int = 100,
) -> dict[str, Any]:
    """Pull recent predictions from DB, extract feature values, run KS test."""
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.timestamp.desc())
        .limit(reference_window + current_window)
        .all()
    )
    values = [
        float(log.features.get(feature_name, 0))
        for log in logs
        if log.features and feature_name in log.features
    ]
    if len(values) < reference_window + current_window:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "reason": "insufficient_data"}
    reference = values[current_window:]
    current = values[:current_window]
    return compute_drift(reference, current)
