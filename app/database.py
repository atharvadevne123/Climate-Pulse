"""SQLAlchemy models and session management."""
from __future__ import annotations

import logging
import os
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./climate_pulse.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
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
    Base.metadata.create_all(bind=engine)
    logger.info("database.init_db: tables created")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
