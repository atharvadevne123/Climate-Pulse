"""SQLAlchemy models and session management."""

from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./climate_pulse.db")
_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = (
    create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if _is_sqlite else {},
        pool_pre_ping=True,
        # Connection pool settings for PostgreSQL; ignored by SQLite
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
    )
    if not _is_sqlite
    else create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    correlation_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    station_id = Column(String, index=True)
    features = Column(JSON)
    predicted_temp = Column(Float)
    predicted_precip = Column(Float)
    extreme_event_prob = Column(Float)
    model_version = Column(String)


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    feature_name = Column(String, index=True)
    ks_statistic = Column(Float)
    p_value = Column(Float)
    drift_detected = Column(Integer)


class WeatherObservation(Base):
    __tablename__ = "weather_observations"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    temperature = Column(Float)
    precipitation = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    wind_speed = Column(Float)
    cloud_cover = Column(Float)


def init_db() -> None:
    """Create all ORM tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("database.init_db: tables created")


def get_db():
    """Yield a SQLAlchemy session; close it after the request completes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def count_predictions(db) -> int:
    """Return total number of persisted prediction log records.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Integer count of all rows in prediction_logs.
    """
    return db.query(PredictionLog).count()


def count_drift_reports(db) -> int:
    """Return total number of persisted drift report records.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Integer count of all rows in drift_reports.
    """
    return db.query(DriftReport).count()


def get_predictions_by_station(db, station_id: str, limit: int = 100) -> list[PredictionLog]:
    """Fetch prediction logs for a specific station, ordered newest-first.

    Args:
        db: Active SQLAlchemy session.
        station_id: Station identifier to filter by.
        limit: Maximum number of records (default 100).

    Returns:
        List of PredictionLog ORM instances.
    """
    return (
        db.query(PredictionLog)
        .filter(PredictionLog.station_id == station_id)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )
