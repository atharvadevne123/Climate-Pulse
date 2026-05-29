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
