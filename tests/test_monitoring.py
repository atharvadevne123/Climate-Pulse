"""Drift detection and monitoring tests."""
from __future__ import annotations

import pytest

from app.monitoring import compute_drift, get_recent_predictions, log_drift_report, log_prediction


class TestComputeDrift:
    def test_no_drift_identical(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = compute_drift(data, data)
        assert result["drift_detected"] is False

    def test_drift_detected_large_shift(self):
        ref = list(range(50))
        cur = [x + 200 for x in range(50)]
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True

    def test_insufficient_data_returns_safe_defaults(self):
        result = compute_drift([1.0, 2.0], [3.0, 4.0])
        assert result["drift_detected"] is False
        assert result["p_value"] == 1.0

    def test_returns_ks_statistic(self):
        ref = [float(i) for i in range(30)]
        cur = [float(i) for i in range(30)]
        result = compute_drift(ref, cur)
        assert "ks_statistic" in result
        assert 0.0 <= result["ks_statistic"] <= 1.0

    @pytest.mark.parametrize("shift", [50, 100, 200])
    def test_larger_shift_detects_drift(self, shift):
        ref = list(range(50))
        cur = [x + shift for x in range(50)]
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True


class TestLogPrediction:
    def test_log_prediction_creates_record(self, db):
        record = log_prediction(
            db=db,
            correlation_id="corr-test-001",
            station_id="STATION_TEST",
            features={"temperature": 20.0},
            predictions={"predicted_temp": 21.0, "predicted_precip": 2.0, "extreme_event_prob": 0.1},
            model_version="1.0.0",
        )
        assert record.id is not None
        assert record.correlation_id == "corr-test-001"

    def test_log_prediction_station_stored(self, db):
        record = log_prediction(
            db=db,
            correlation_id="corr-test-002",
            station_id="STATION_XYZ",
            features={"temperature": 15.0},
            predictions={"predicted_temp": 16.0, "predicted_precip": 1.0, "extreme_event_prob": 0.05},
            model_version="1.0.0",
        )
        assert record.station_id == "STATION_XYZ"


class TestLogDriftReport:
    def test_log_drift_report_creates_record(self, db):
        drift = {"ks_statistic": 0.35, "p_value": 0.01, "drift_detected": True}
        report = log_drift_report(db, "temperature", drift)
        assert report.id is not None
        assert report.feature_name == "temperature"
        assert report.drift_detected == 1


class TestGetRecentPredictions:
    def test_returns_list(self, db):
        result = get_recent_predictions(db, limit=10)
        assert isinstance(result, list)

    def test_limit_respected(self, db):
        for i in range(5):
            log_prediction(
                db=db,
                correlation_id=f"corr-limit-{i}",
                station_id="LIMIT_TEST",
                features={"temperature": float(i)},
                predictions={"predicted_temp": float(i + 1), "predicted_precip": 0.0, "extreme_event_prob": 0.0},
                model_version="1.0.0",
            )
        result = get_recent_predictions(db, limit=3)
        assert len(result) <= 3
