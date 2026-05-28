"""Tests for SQLAlchemy database models and session management."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.database import DriftReport, PredictionLog, WeatherObservation


class TestPredictionLogModel:
    def test_create_prediction_log(self, db):
        record = PredictionLog(
            correlation_id="corr-001",
            timestamp=datetime.now(UTC),
            station_id="STATION_001",
            features={"temperature": 22.5, "humidity": 65.0},
            predicted_temp=23.1,
            predicted_precip=2.0,
            extreme_event_prob=0.05,
            model_version="1.0.0",
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        assert record.id is not None
        assert record.correlation_id == "corr-001"
        assert record.station_id == "STATION_001"

    def test_prediction_log_features_as_json(self, db):
        features = {"temperature": 20.0, "humidity": 60.0, "pressure": 1013.0}
        record = PredictionLog(
            correlation_id="corr-002",
            timestamp=datetime.now(UTC),
            station_id="S1",
            features=features,
            predicted_temp=21.0,
            predicted_precip=1.0,
            extreme_event_prob=0.02,
            model_version="1.0.0",
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        assert record.features["temperature"] == 20.0

    @pytest.mark.parametrize("station_id", ["AWS-001", "BERLIN_TEGEL", "JFK_01", "S99"])
    def test_multiple_station_ids(self, db, station_id):
        record = PredictionLog(
            correlation_id=f"corr-{station_id}",
            timestamp=datetime.now(UTC),
            station_id=station_id,
            features={},
            predicted_temp=15.0,
            predicted_precip=0.0,
            extreme_event_prob=0.01,
            model_version="1.0.0",
        )
        db.add(record)
        db.commit()
        assert record.station_id == station_id

    def test_query_prediction_logs(self, db):
        for i in range(3):
            db.add(PredictionLog(
                correlation_id=f"c{i}",
                timestamp=datetime.now(UTC),
                station_id="S1",
                features={},
                predicted_temp=float(i),
                predicted_precip=0.0,
                extreme_event_prob=0.0,
                model_version="1.0.0",
            ))
        db.commit()
        logs = db.query(PredictionLog).filter_by(station_id="S1").all()
        assert len(logs) == 3


class TestDriftReportModel:
    def test_create_drift_report(self, db):
        report = DriftReport(
            timestamp=datetime.now(UTC),
            feature_name="temperature",
            ks_statistic=0.12,
            p_value=0.03,
            drift_detected=1,
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        assert report.id is not None
        assert report.drift_detected == 1

    @pytest.mark.parametrize("feature", ["temperature", "humidity", "pressure", "wind_speed"])
    def test_drift_report_all_features(self, db, feature):
        report = DriftReport(
            timestamp=datetime.now(UTC),
            feature_name=feature,
            ks_statistic=0.05,
            p_value=0.20,
            drift_detected=0,
        )
        db.add(report)
        db.commit()
        assert report.feature_name == feature


class TestWeatherObservationModel:
    def test_create_weather_observation(self, db):
        obs = WeatherObservation(
            station_id="BERLIN_001",
            timestamp=datetime.now(UTC),
            temperature=18.5,
            precipitation=1.2,
            humidity=72.0,
            pressure=1015.0,
            wind_speed=12.0,
            cloud_cover=55.0,
        )
        db.add(obs)
        db.commit()
        db.refresh(obs)
        assert obs.id is not None
        assert obs.station_id == "BERLIN_001"
