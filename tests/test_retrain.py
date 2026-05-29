"""Tests for the /api/v1/retrain endpoint and model retraining flow."""

from __future__ import annotations


class TestRetrainEndpoint:
    def test_retrain_returns_200(self, client):
        resp = client.post("/api/v1/retrain")
        assert resp.status_code == 200

    def test_retrain_response_has_status(self, client):
        data = client.post("/api/v1/retrain").json()
        assert data.get("status") == "retrained"

    def test_retrain_returns_metrics(self, client):
        data = client.post("/api/v1/retrain").json()
        assert "temp_r2_mean" in data
        assert "extreme_auc_mean" in data

    def test_retrain_invalidates_metrics_cache(self, client):
        m1 = client.get("/api/v1/metrics").json()
        client.post("/api/v1/retrain")
        m2 = client.get("/api/v1/metrics").json()
        assert m2["model_version"] == m1["model_version"]

    def test_retrain_n_training_samples_positive(self, client):
        data = client.post("/api/v1/retrain").json()
        assert data.get("n_training_samples", 0) > 0


class TestRetrainMetrics:
    def test_retrain_metrics_n_training_samples(self, client):
        data = client.post("/api/v1/retrain").json()
        assert data.get("n_training_samples", 0) >= 100

    def test_retrain_metrics_n_features_positive(self, client):
        data = client.post("/api/v1/retrain").json()
        assert data.get("n_features", 0) > 0

    def test_retrain_temp_r2_std_non_negative(self, client):
        data = client.post("/api/v1/retrain").json()
        assert data.get("temp_r2_std", -1) >= 0

    def test_retrain_auc_mean_at_least_half(self, client):
        data = client.post("/api/v1/retrain").json()
        assert data.get("extreme_auc_mean", 0) >= 0.5

    def test_retrain_model_version_string(self, client):
        data = client.post("/api/v1/retrain").json()
        assert isinstance(data.get("model_version"), str)


class TestRetrainModelFreshness:
    def test_model_freshness_after_retrain(self, client):
        client.post("/api/v1/retrain")
        data = client.get("/api/v1/model/freshness").json()
        assert data.get("stale_threshold_hours") == 24

    def test_predict_still_works_after_retrain(self, client, sample_weather_payload):
        client.post("/api/v1/retrain")
        resp = client.post("/api/v1/predict", json=sample_weather_payload)
        assert resp.status_code == 200
        assert "predicted_temp" in resp.json()

    def test_metrics_accessible_after_retrain(self, client):
        client.post("/api/v1/retrain")
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
        assert resp.json()["n_training_samples"] > 0
