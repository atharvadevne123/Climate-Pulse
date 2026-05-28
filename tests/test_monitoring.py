"""Drift detection and monitoring tests."""
from __future__ import annotations

import pytest

from app.monitoring import (
    compute_drift,
    get_prediction_by_correlation_id,
    get_recent_predictions,
    log_drift_report,
    log_prediction,
)


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

    @pytest.mark.parametrize("limit", [1, 5, 10, 50])
    def test_limit_values(self, db, limit):
        result = get_recent_predictions(db, limit=limit)
        assert len(result) <= limit

    def test_returns_most_recent_first(self, db):
        for i in range(3):
            log_prediction(
                db=db,
                correlation_id=f"order-{i}",
                station_id="ORDER_TEST",
                features={"temperature": float(i)},
                predictions={"predicted_temp": float(i), "predicted_precip": 0.0, "extreme_event_prob": 0.0},
                model_version="1.0.0",
            )
        result = get_recent_predictions(db, limit=10)
        ids = [r.id for r in result]
        assert ids == sorted(ids, reverse=True)


class TestComputeDriftEdgeCases:
    @pytest.mark.parametrize("size", [5, 10, 30, 100])
    def test_identical_distributions_no_drift(self, size):
        data = [float(i) for i in range(size)]
        result = compute_drift(data, data)
        assert result["drift_detected"] is False

    def test_ks_statistic_bounded(self):
        ref = [1.0] * 20
        cur = [2.0] * 20
        result = compute_drift(ref, cur)
        assert 0.0 <= result["ks_statistic"] <= 1.0

    def test_p_value_bounded(self):
        ref = [float(i) for i in range(20)]
        cur = [float(i) for i in range(20)]
        result = compute_drift(ref, cur)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_exactly_five_elements_valid(self):
        result = compute_drift([1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0])
        assert "ks_statistic" in result
        assert "reason" not in result

    def test_four_elements_insufficient(self):
        result = compute_drift([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
        assert result["drift_detected"] is False
        assert result.get("reason") == "insufficient_data"


class TestGetPredictionByCorrelationId:
    def test_returns_none_for_unknown_id(self, db):
        result = get_prediction_by_correlation_id(db, "nonexistent-id-xyz-000")
        assert result is None

    def test_returns_record_for_known_id(self, db):
        log_prediction(
            db=db,
            correlation_id="lookup-corr-001",
            station_id="LOOKUP_STATION",
            features={"temperature": 22.0},
            predictions={"predicted_temp": 23.0, "predicted_precip": 1.0, "extreme_event_prob": 0.05},
            model_version="1.0.0",
        )
        result = get_prediction_by_correlation_id(db, "lookup-corr-001")
        assert result is not None
        assert result.correlation_id == "lookup-corr-001"

    def test_returns_correct_station(self, db):
        log_prediction(
            db=db,
            correlation_id="lookup-corr-002",
            station_id="STATION_ALPHA",
            features={"temperature": 18.0},
            predictions={"predicted_temp": 19.0, "predicted_precip": 0.5, "extreme_event_prob": 0.02},
            model_version="1.0.0",
        )
        result = get_prediction_by_correlation_id(db, "lookup-corr-002")
        assert result.station_id == "STATION_ALPHA"

    @pytest.mark.parametrize("corr_id", ["abc-001", "def-002", "ghi-003"])
    def test_lookup_various_ids(self, db, corr_id):
        log_prediction(
            db=db,
            correlation_id=corr_id,
            station_id="PARAM_STATION",
            features={"temperature": 10.0},
            predictions={"predicted_temp": 11.0, "predicted_precip": 0.0, "extreme_event_prob": 0.0},
            model_version="1.0.0",
        )
        result = get_prediction_by_correlation_id(db, corr_id)
        assert result is not None
        assert result.correlation_id == corr_id
