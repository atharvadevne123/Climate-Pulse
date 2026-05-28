"""Tests for new API endpoints: /readyz, /version, /drift/history."""
from __future__ import annotations

import pytest


class TestReadyzEndpoint:
    def test_readyz_returns_200(self, client):
        resp = client.get("/api/v1/readyz")
        assert resp.status_code == 200

    def test_readyz_status_field(self, client):
        data = client.get("/api/v1/readyz").json()
        assert data.get("status") == "ready"


class TestVersionEndpoint:
    def test_version_returns_200(self, client):
        assert client.get("/api/v1/version").status_code == 200

    def test_version_has_api_version(self, client):
        data = client.get("/api/v1/version").json()
        assert "api_version" in data

    def test_version_has_model_version(self, client):
        data = client.get("/api/v1/version").json()
        assert "model_version" in data

    def test_version_strings(self, client):
        data = client.get("/api/v1/version").json()
        assert isinstance(data["api_version"], str)
        assert isinstance(data["model_version"], str)


class TestDriftHistoryEndpoint:
    def test_drift_history_returns_list(self, client):
        resp = client.get("/api/v1/drift/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_drift_history_limit_param(self, client):
        resp = client.get("/api/v1/drift/history?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()) <= 5

    def test_drift_history_limit_too_high(self, client):
        resp = client.get("/api/v1/drift/history?limit=201")
        assert resp.status_code == 400

    @pytest.mark.parametrize("limit", [1, 5, 10, 20])
    def test_drift_history_valid_limits(self, client, limit):
        resp = client.get(f"/api/v1/drift/history?limit={limit}")
        assert resp.status_code == 200


class TestStationHistoryEndpoint:
    def test_station_history_returns_list(self, client, sample_weather_payload):
        client.post("/api/v1/predict", json=sample_weather_payload)
        resp = client.get(f"/api/v1/stations/{sample_weather_payload['station_id']}/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_station_history_unknown_station_empty(self, client):
        resp = client.get("/api/v1/stations/NONEXISTENT_XYZ_999/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_station_history_limit_too_high(self, client):
        resp = client.get("/api/v1/stations/S1/history?limit=201")
        assert resp.status_code == 400

    @pytest.mark.parametrize("limit", [1, 5, 10])
    def test_station_history_valid_limits(self, client, limit):
        resp = client.get(f"/api/v1/stations/S1/history?limit={limit}")
        assert resp.status_code == 200

    def test_station_history_record_fields(self, client, sample_weather_payload):
        client.post("/api/v1/predict", json=sample_weather_payload)
        records = client.get(f"/api/v1/stations/{sample_weather_payload['station_id']}/history?limit=1").json()
        if records:
            record = records[0]
            assert "id" in record
            assert "station_id" in record
            assert "predicted_temp" in record
            assert "timestamp" in record


class TestTelemetryEndpoint:
    def test_telemetry_returns_200(self, client):
        resp = client.get("/api/v1/telemetry")
        assert resp.status_code == 200

    def test_telemetry_returns_dict(self, client):
        data = client.get("/api/v1/telemetry").json()
        assert isinstance(data, dict)

    def test_telemetry_has_counters_after_predict(self, client, sample_weather_payload):
        client.post("/api/v1/predict", json=sample_weather_payload)
        data = client.get("/api/v1/telemetry").json()
        assert "counters" in data

    def test_telemetry_predictions_counter_increments(self, client, sample_weather_payload):
        before = client.get("/api/v1/telemetry").json().get("counters", {}).get("predictions.total", 0)
        client.post("/api/v1/predict", json=sample_weather_payload)
        after = client.get("/api/v1/telemetry").json()["counters"]["predictions.total"]
        assert after > before
