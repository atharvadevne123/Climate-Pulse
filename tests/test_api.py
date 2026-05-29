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


class TestModelInfo:
    def test_model_info_returns_200(self, client):
        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200

    def test_model_info_has_version(self, client):
        data = client.get("/api/v1/model/info").json()
        assert "model_version" in data

    def test_model_info_has_pipeline_stages(self, client):
        data = client.get("/api/v1/model/info").json()
        assert "pipeline_stages" in data
        assert isinstance(data["pipeline_stages"], list)

    def test_model_info_has_input_features(self, client):
        data = client.get("/api/v1/model/info").json()
        assert "input_features" in data
        assert "temperature" in data["input_features"]

    def test_model_info_has_is_trained_flag(self, client):
        data = client.get("/api/v1/model/info").json()
        assert "is_trained" in data
        assert isinstance(data["is_trained"], bool)


class TestStationStats:
    def test_empty_station_returns_200(self, client):
        resp = client.get("/api/v1/stations/UNKNOWN_XYZ/stats")
        assert resp.status_code == 200

    def test_empty_station_has_zero_count(self, client):
        data = client.get("/api/v1/stations/NOPREDICTIONS/stats").json()
        assert data["count"] == 0

    def test_station_with_predictions_has_stats(self, client, sample_weather_payload):
        client.post("/api/v1/predict", json={**sample_weather_payload, "station_id": "STATS_TEST"})
        data = client.get("/api/v1/stations/STATS_TEST/stats").json()
        assert data["count"] >= 1


class TestCacheStatsEndpoint:
    def test_cache_stats_returns_200(self, client):
        resp = client.get("/api/v1/cache/stats")
        assert resp.status_code == 200

    def test_cache_stats_has_required_fields(self, client):
        data = client.get("/api/v1/cache/stats").json()
        assert "size" in data
        assert "total_hits" in data
        assert "hit_rate" in data


class TestDriftSummaryEndpoint:
    def test_drift_summary_returns_200(self, client):
        resp = client.get("/api/v1/drift/summary")
        assert resp.status_code == 200

    def test_drift_summary_has_counts_field(self, client):
        data = client.get("/api/v1/drift/summary").json()
        assert "drift_counts_by_feature" in data
        assert isinstance(data["drift_counts_by_feature"], dict)


class TestVersionEndpoint:
    def test_version_returns_200(self, client):
        resp = client.get("/api/v1/version")
        assert resp.status_code == 200

    def test_version_has_api_version(self, client):
        data = client.get("/api/v1/version").json()
        assert "api_version" in data

    def test_version_has_model_version(self, client):
        data = client.get("/api/v1/version").json()
        assert "model_version" in data


class TestRetrain:
    def test_retrain_returns_200(self, client):
        resp = client.post("/api/v1/retrain")
        assert resp.status_code == 200

    def test_retrain_status_field(self, client):
        data = client.post("/api/v1/retrain").json()
        assert data.get("status") == "retrained"

    def test_retrain_returns_model_version(self, client):
        data = client.post("/api/v1/retrain").json()
        assert "model_version" in data
        assert isinstance(data["model_version"], str)

    def test_retrain_metrics_present(self, client):
        data = client.post("/api/v1/retrain").json()
        assert "metrics" in data


class TestPredictResponseHeaders:
    def test_correlation_id_header_in_response(self, client, sample_weather_payload):
        resp = client.post("/api/v1/predict", json=sample_weather_payload)
        assert "X-Correlation-ID" in resp.headers

    def test_custom_correlation_id_echoed(self, client, sample_weather_payload):
        resp = client.post(
            "/api/v1/predict",
            json=sample_weather_payload,
            headers={"X-Correlation-ID": "echo-test-42"},
        )
        assert resp.headers.get("X-Correlation-ID") == "echo-test-42"

    def test_predict_returns_station_id(self, client, sample_weather_payload):
        data = client.post("/api/v1/predict", json=sample_weather_payload).json()
        assert data["station_id"] == sample_weather_payload["station_id"]
