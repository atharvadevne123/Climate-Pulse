"""Extended API endpoint tests for Climate-Pulse — edge cases and parametrized scenarios."""
from __future__ import annotations

import pytest


class TestPredictParametrized:
    @pytest.mark.parametrize("temperature", [-89.0, 0.0, 15.0, 35.0, 59.0])
    def test_predict_valid_temperature_range(self, client, sample_weather_payload, temperature):
        payload = {**sample_weather_payload, "temperature": temperature}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    @pytest.mark.parametrize("temperature", [-91.0, 61.0, 999.0])
    def test_predict_invalid_temperature_rejected(self, client, sample_weather_payload, temperature):
        payload = {**sample_weather_payload, "temperature": temperature}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    @pytest.mark.parametrize("humidity", [0.0, 50.0, 100.0])
    def test_predict_valid_humidity_range(self, client, sample_weather_payload, humidity):
        payload = {**sample_weather_payload, "humidity": humidity}
        assert client.post("/api/v1/predict", json=payload).status_code == 200

    @pytest.mark.parametrize("month", [1.0, 6.0, 12.0])
    def test_predict_valid_months(self, client, sample_weather_payload, month):
        payload = {**sample_weather_payload, "month": month}
        assert client.post("/api/v1/predict", json=payload).status_code == 200

    @pytest.mark.parametrize("month", [0.0, 13.0, -1.0])
    def test_predict_invalid_months_rejected(self, client, sample_weather_payload, month):
        payload = {**sample_weather_payload, "month": month}
        assert client.post("/api/v1/predict", json=payload).status_code == 422

    def test_predict_model_version_in_response(self, client, sample_weather_payload):
        data = client.post("/api/v1/predict", json=sample_weather_payload).json()
        assert "model_version" in data
        assert isinstance(data["model_version"], str)

    def test_predict_station_id_echoed(self, client, sample_weather_payload):
        data = client.post("/api/v1/predict", json=sample_weather_payload).json()
        assert data["station_id"] == sample_weather_payload["station_id"]

    def test_predict_auto_generates_correlation_id(self, client, sample_weather_payload):
        resp = client.post("/api/v1/predict", json=sample_weather_payload)
        data = resp.json()
        assert "correlation_id" in data
        assert len(data["correlation_id"]) > 0

    def test_predict_correlation_id_echoed_in_header(self, client, sample_weather_payload):
        resp = client.post(
            "/api/v1/predict",
            json=sample_weather_payload,
            headers={"X-Correlation-ID": "hdr-test-99"},
        )
        assert resp.headers.get("X-Correlation-ID") == "hdr-test-99"

    def test_predict_wind_speed_zero_valid(self, client, sample_weather_payload):
        payload = {**sample_weather_payload, "wind_speed": 0.0}
        assert client.post("/api/v1/predict", json=payload).status_code == 200

    def test_predict_precipitation_zero_valid(self, client, sample_weather_payload):
        payload = {**sample_weather_payload, "precipitation": 0.0}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["predicted_precip"] >= 0.0


class TestDriftParametrized:
    @pytest.mark.parametrize("size", [5, 10, 50, 100])
    def test_drift_check_various_input_sizes(self, client, size):
        data = {"reference": list(range(size)), "current": list(range(size))}
        resp = client.post("/api/v1/drift", json=data)
        assert resp.status_code == 200
        assert "drift_detected" in resp.json()

    def test_drift_check_missing_reference_422(self, client):
        resp = client.post("/api/v1/drift", json={"current": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert resp.status_code == 422

    def test_drift_check_too_short_422(self, client):
        resp = client.post("/api/v1/drift", json={"reference": [1.0, 2.0], "current": [3.0, 4.0]})
        assert resp.status_code == 422

    @pytest.mark.parametrize("feature", ["temperature", "precipitation", "humidity", "pressure"])
    def test_drift_feature_valid_names(self, client, feature):
        resp = client.get(f"/api/v1/drift/features?feature={feature}")
        assert resp.status_code in (200, 400)

    def test_drift_ks_statistic_in_range(self, client):
        ref = [float(i) for i in range(20)]
        cur = [float(i) + 0.5 for i in range(20)]
        data = client.post("/api/v1/drift", json={"reference": ref, "current": cur}).json()
        assert 0.0 <= data["ks_statistic"] <= 1.0

    def test_drift_p_value_in_range(self, client):
        ref = [float(i) for i in range(20)]
        cur = [float(i) for i in range(20)]
        data = client.post("/api/v1/drift", json={"reference": ref, "current": cur}).json()
        assert 0.0 <= data["p_value"] <= 1.0


class TestRecentPredictionsParametrized:
    @pytest.mark.parametrize("limit", [1, 5, 10, 20, 100])
    def test_recent_predictions_valid_limits(self, client, limit):
        resp = client.get(f"/api/v1/predictions/recent?limit={limit}")
        assert resp.status_code == 200

    def test_recent_predictions_default_limit(self, client):
        resp = client.get("/api/v1/predictions/recent")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_recent_predictions_returns_expected_fields(self, client, sample_weather_payload):
        client.post("/api/v1/predict", json=sample_weather_payload)
        records = client.get("/api/v1/predictions/recent?limit=1").json()
        if records:
            record = records[0]
            assert "id" in record
            assert "station_id" in record
            assert "predicted_temp" in record
            assert "timestamp" in record

    def test_recent_predictions_limit_201_rejected(self, client):
        resp = client.get("/api/v1/predictions/recent?limit=201")
        assert resp.status_code == 400


class TestHealthEndpointRobustness:
    def test_health_endpoint_no_auth_required(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_correlation_id_header(self, client):
        resp = client.get("/api/v1/health")
        assert "X-Correlation-ID" in resp.headers

    def test_metrics_version_field(self, client):
        data = client.get("/api/v1/metrics").json()
        assert "model_version" in data

    def test_metrics_sample_count_positive(self, client):
        data = client.get("/api/v1/metrics").json()
        assert data["n_training_samples"] > 0
