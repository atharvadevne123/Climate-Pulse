"""Tests for SQLAlchemy database models and session management."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.database import (
    DriftReport,
    PredictionLog,
    WeatherObservation,
    count_drift_reports,
    count_predictions,
    get_predictions_by_station,
)


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
            db.add(
                PredictionLog(
                    correlation_id=f"c{i}",
                    timestamp=datetime.now(UTC),
                    station_id="S1",
                    features={},
                    predicted_temp=float(i),
                    predicted_precip=0.0,
                    extreme_event_prob=0.0,
                    model_version="1.0.0",
                )
            )
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


class TestDatabaseHelpers:
    def test_count_predictions_empty(self, db):
        assert count_predictions(db) == 0

    def test_count_predictions_after_insert(self, db):
        db.add(
            PredictionLog(
                correlation_id="helper-001",
                timestamp=datetime.now(UTC),
                station_id="H1",
                features={},
                predicted_temp=10.0,
                predicted_precip=0.0,
                extreme_event_prob=0.0,
                model_version="1.0.0",
            )
        )
        db.commit()
        assert count_predictions(db) == 1

    def test_count_drift_reports_empty(self, db):
        assert count_drift_reports(db) == 0

    def test_count_drift_reports_after_insert(self, db):
        db.add(
            DriftReport(
                timestamp=datetime.now(UTC),
                feature_name="temperature",
                ks_statistic=0.1,
                p_value=0.5,
                drift_detected=0,
            )
        )
        db.commit()
        assert count_drift_reports(db) == 1

    def test_get_predictions_by_station_empty(self, db):
        result = get_predictions_by_station(db, "UNKNOWN_STATION")
        assert result == []

    def test_get_predictions_by_station_returns_correct_records(self, db):
        for i in range(3):
            db.add(
                PredictionLog(
                    correlation_id=f"station-{i}",
                    timestamp=datetime.now(UTC),
                    station_id="QUERY_STATION",
                    features={},
                    predicted_temp=float(i),
                    predicted_precip=0.0,
                    extreme_event_prob=0.0,
                    model_version="1.0.0",
                )
            )
        db.commit()
        result = get_predictions_by_station(db, "QUERY_STATION")
        assert len(result) == 3
        assert all(r.station_id == "QUERY_STATION" for r in result)

    def test_get_predictions_by_station_limit(self, db):
        for i in range(5):
            db.add(
                PredictionLog(
                    correlation_id=f"limit-{i}",
                    timestamp=datetime.now(UTC),
                    station_id="LIMIT_STATION",
                    features={},
                    predicted_temp=float(i),
                    predicted_precip=0.0,
                    extreme_event_prob=0.0,
                    model_version="1.0.0",
                )
            )
        db.commit()
        result = get_predictions_by_station(db, "LIMIT_STATION", limit=2)
        assert len(result) == 2


class TestNewDatabaseHelpers:
    def test_get_oldest_prediction_empty(self, db):
        from app.database import get_oldest_prediction

        assert get_oldest_prediction(db) is None

    def test_get_oldest_prediction_returns_first(self, db):
        from datetime import UTC, datetime, timedelta

        from app.database import get_oldest_prediction

        for i in range(3):
            db.add(
                PredictionLog(
                    correlation_id=f"oldest-{i}",
                    timestamp=datetime.now(UTC) - timedelta(seconds=10 - i),
                    station_id="OLD_STATION",
                    features={},
                    predicted_temp=float(i),
                    predicted_precip=0.0,
                    extreme_event_prob=0.0,
                    model_version="1.0.0",
                )
            )
        db.commit()
        result = get_oldest_prediction(db)
        assert result is not None
        assert result.correlation_id == "oldest-0"

    def test_get_prediction_count_by_station_zero(self, db):
        from app.database import get_prediction_count_by_station

        assert get_prediction_count_by_station(db, "GHOST_STATION") == 0

    def test_get_prediction_count_by_station_counts_correctly(self, db):
        from app.database import get_prediction_count_by_station

        for i in range(4):
            db.add(
                PredictionLog(
                    correlation_id=f"cnt-{i}",
                    timestamp=datetime.now(UTC),
                    station_id="COUNT_STATION",
                    features={},
                    predicted_temp=float(i),
                    predicted_precip=0.0,
                    extreme_event_prob=0.0,
                    model_version="1.0.0",
                )
            )
        db.commit()
        assert get_prediction_count_by_station(db, "COUNT_STATION") == 4

    @pytest.mark.parametrize("count", [1, 3, 7])
    def test_get_prediction_count_by_station_parametrized(self, db, count):
        from app.database import get_prediction_count_by_station

        station = f"PARAM_ST_{count}"
        for i in range(count):
            db.add(
                PredictionLog(
                    correlation_id=f"pcount-{count}-{i}",
                    timestamp=datetime.now(UTC),
                    station_id=station,
                    features={},
                    predicted_temp=float(i),
                    predicted_precip=0.0,
                    extreme_event_prob=0.0,
                    model_version="1.0.0",
                )
            )
        db.commit()
        assert get_prediction_count_by_station(db, station) == count


class TestGetPredictionsBetween:
    def test_returns_empty_when_none_in_range(self, db):
        from datetime import timedelta

        from app.database import get_predictions_between

        future = datetime.now(UTC) + timedelta(hours=1)
        result = get_predictions_between(db, future, future + timedelta(hours=1))
        assert result == []

    def test_returns_records_in_window(self, db):
        from datetime import timedelta

        from app.database import get_predictions_between

        now = datetime.now(UTC)
        for i in range(3):
            db.add(
                PredictionLog(
                    correlation_id=f"between-{i}",
                    timestamp=now + timedelta(seconds=i),
                    station_id="BTW_ST",
                    features={},
                    predicted_temp=float(i),
                    predicted_precip=0.0,
                    extreme_event_prob=0.0,
                    model_version="1.0.0",
                )
            )
        db.commit()
        end = now + timedelta(seconds=10)
        result = get_predictions_between(db, now - timedelta(seconds=1), end)
        assert len(result) == 3

    def test_excludes_records_outside_window(self, db):
        from datetime import timedelta

        from app.database import get_predictions_between

        past = datetime.now(UTC) - timedelta(hours=2)
        db.add(
            PredictionLog(
                correlation_id="outside-window",
                timestamp=past,
                station_id="OUT_ST",
                features={},
                predicted_temp=10.0,
                predicted_precip=0.0,
                extreme_event_prob=0.0,
                model_version="1.0.0",
            )
        )
        db.commit()
        recent_start = datetime.now(UTC) - timedelta(minutes=30)
        recent_end = datetime.now(UTC)
        result = get_predictions_between(db, recent_start, recent_end)
        assert all(r.correlation_id != "outside-window" for r in result)
