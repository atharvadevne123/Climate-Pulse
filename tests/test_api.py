"""API endpoint tests for Climate-Pulse."""

from __future__ import annotations

import pytest


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_has_version(self, client):
        resp = client.get("/api/v1/health")
        assert "version" in resp.json()


class TestPredict:
    def test_predict_returns_200(self, client, sample_weather_payload):
        resp = client.post("/api/v1/predict", json=sample_weather_payload)
        assert resp.status_code == 200

    def test_predict_response_fields(self, client, sample_weather_payload):
        resp = client.post("/api/v1/predict", json=sample_weather_payload)
        data = resp.json()
        assert "predicted_temp" in data
        assert "predicted_precip" in data
        assert "extreme_event_prob" in data
        assert "correlation_id" in data

    def test_predict_precip_non_negative(self, client, sample_weather_payload):
        resp = client.post("/api/v1/predict", json=sample_weather_payload)
        assert resp.json()["predicted_precip"] >= 0

    def test_predict_extreme_prob_in_range(self, client, sample_weather_payload):
        resp = client.post("/api/v1/predict", json=sample_weather_payload)
        p = resp.json()["extreme_event_prob"]
        assert 0.0 <= p <= 1.0

    @pytest.mark.parametrize("field", ["temperature", "humidity", "pressure"])
    def test_predict_missing_field_422(self, client, sample_weather_payload, field):
        payload = {k: v for k, v in sample_weather_payload.items() if k != field}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_temperature_422(self, client, sample_weather_payload):
        payload = {**sample_weather_payload, "temperature": 200}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_empty_station_id_422(self, client, sample_weather_payload):
        payload = {**sample_weather_payload, "station_id": "   "}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_stores_correlation_id(self, client, sample_weather_payload):
        resp = client.post(
            "/api/v1/predict",
            json=sample_weather_payload,
            headers={"X-Correlation-ID": "test-corr-123"},
        )
        assert resp.json()["correlation_id"] == "test-corr-123"


class TestMetrics:
    def test_metrics_returns_200(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_has_required_fields(self, client):
        data = client.get("/api/v1/metrics").json()
        assert "temp_r2_mean" in data
        assert "extreme_auc_mean" in data


class TestDrift:
    def test_drift_no_drift(self, client):
        ref = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        cur = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
        resp = client.post("/api/v1/drift", json={"reference": ref, "current": cur})
        assert resp.status_code == 200
        assert "ks_statistic" in resp.json()

    def test_drift_detects_shift(self, client):
        ref = list(range(50))
        cur = [x + 100 for x in range(50)]
        resp = client.post("/api/v1/drift", json={"reference": ref, "current": cur})
        assert resp.json()["drift_detected"] is True

    def test_drift_feature_invalid_feature(self, client):
        resp = client.get("/api/v1/drift/features?feature=invalid_xyz")
        assert resp.status_code == 400


class TestRecentPredictions:
    def test_recent_predictions_returns_list(self, client, sample_weather_payload):
        client.post("/api/v1/predict", json=sample_weather_payload)
        resp = client.get("/api/v1/predictions/recent?limit=5")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_recent_predictions_limit_too_high(self, client):
        resp = client.get("/api/v1/predictions/recent?limit=9999")
        assert resp.status_code == 400
