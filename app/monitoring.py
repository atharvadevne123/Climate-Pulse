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
    return db.query(PredictionLog).filter(PredictionLog.correlation_id == correlation_id).first()


def get_drift_history(db: Session, limit: int = 100) -> list[DriftReport]:
    """Fetch the most recent drift reports ordered by timestamp desc.

    Args:
        db: Database session.
        limit: Maximum number of records to return.

    Returns:
        List of DriftReport ORM instances.
    """
    return db.query(DriftReport).order_by(DriftReport.timestamp.desc()).limit(limit).all()


def get_station_stats(db: Session, station_id: str) -> dict[str, Any]:
    """Return aggregate prediction statistics for a single station.

    Args:
        db: Active SQLAlchemy session.
        station_id: Station identifier to aggregate.

    Returns:
        Dict with count, avg/min/max for temp/precip/extreme_prob.
    """
    logs = db.query(PredictionLog).filter(PredictionLog.station_id == station_id).all()
    if not logs:
        return {"station_id": station_id, "count": 0}

    temps = [rec.predicted_temp for rec in logs if rec.predicted_temp is not None]
    precips = [rec.predicted_precip for rec in logs if rec.predicted_precip is not None]
    extremes = [rec.extreme_event_prob for rec in logs if rec.extreme_event_prob is not None]

    def _stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"avg": 0.0, "min": 0.0, "max": 0.0}
        return {
            "avg": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    return {
        "station_id": station_id,
        "count": len(logs),
        "temperature": _stats(temps),
        "precipitation": _stats(precips),
        "extreme_event_prob": _stats(extremes),
    }


def get_drift_count_by_feature(db: Session) -> dict[str, int]:
    """Return number of drift events detected per feature name.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Dict mapping feature name to count of drift-detected reports.
    """
    reports = db.query(DriftReport).filter(DriftReport.drift_detected == 1).all()
    counts: dict[str, int] = {}
    for r in reports:
        counts[r.feature_name] = counts.get(r.feature_name, 0) + 1
    return counts


def purge_old_predictions(db: Session, keep_latest: int = 10000) -> int:
    """Delete prediction logs beyond the most recent *keep_latest* records.

    Args:
        db: Active SQLAlchemy session.
        keep_latest: Number of most-recent records to retain (default 10 000).

    Returns:
        Number of rows deleted.
    """
    total = db.query(PredictionLog).count()
    if total <= keep_latest:
        return 0
    cutoff_id_row = (
        db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).offset(keep_latest - 1).limit(1).first()
    )
    if cutoff_id_row is None:
        return 0
    deleted = db.query(PredictionLog).filter(PredictionLog.id < cutoff_id_row.id).delete(synchronize_session=False)
    db.commit()
    logger.info("monitoring.purge_old_predictions: deleted=%d", deleted)
    return deleted


def compute_feature_drift_from_db(
    db: Session,
    feature_name: str,
    reference_window: int = 500,
    current_window: int = 100,
) -> dict[str, Any]:
    """Pull recent predictions from DB, extract feature values, run KS test."""
    logs = (
        db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(reference_window + current_window).all()
    )
    values = [float(log.features.get(feature_name, 0)) for log in logs if log.features and feature_name in log.features]
    if len(values) < reference_window + current_window:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "reason": "insufficient_data"}
    reference = values[current_window:]
    current = values[:current_window]
    return compute_drift(reference, current)
