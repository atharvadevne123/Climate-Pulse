"""Model training and prediction tests."""
from __future__ import annotations

import pandas as pd
import pytest

from app.model import _generate_synthetic_training_data, get_metrics, predict, train_models


@pytest.fixture(scope="module")
def trained_metrics():
    return train_models()


class TestSyntheticData:
    def test_generate_returns_correct_shape(self):
        X, y_temp, y_precip, y_extreme = _generate_synthetic_training_data(n=200)
        assert len(X) == 200
        assert len(y_temp) == 200
        assert len(y_precip) == 200
        assert len(y_extreme) == 200

    def test_temperature_in_range(self):
        X, _, _, _ = _generate_synthetic_training_data(n=500)
        assert X["temperature"].between(-30, 50).all()

    def test_precipitation_non_negative(self):
        X, _, _, _ = _generate_synthetic_training_data(n=500)
        assert (X["precipitation"] >= 0).all()

    def test_extreme_labels_binary(self):
        _, _, _, y_extreme = _generate_synthetic_training_data(n=500)
        assert set(y_extreme.unique()).issubset({0, 1})


class TestTrainModels:
    def test_returns_dict(self, trained_metrics):
        assert isinstance(trained_metrics, dict)

    def test_has_r2_metrics(self, trained_metrics):
        assert "temp_r2_mean" in trained_metrics
        assert "precip_r2_mean" in trained_metrics

    def test_has_auc_metrics(self, trained_metrics):
        assert "extreme_auc_mean" in trained_metrics

    @pytest.mark.parametrize("key", ["temp_r2_mean", "precip_r2_mean"])
    def test_r2_reasonable(self, trained_metrics, key):
        # R² should be positive for a learnable problem
        assert trained_metrics[key] > -1.0

    def test_auc_above_random(self, trained_metrics):
        assert trained_metrics["extreme_auc_mean"] >= 0.5

    def test_n_features_gt_zero(self, trained_metrics):
        assert trained_metrics["n_features"] > 0


class TestPredict:
    def test_predict_returns_dict(self):
        df = pd.DataFrame(
            [
                {
                    "temperature": 20.0,
                    "precipitation": 2.0,
                    "humidity": 60.0,
                    "pressure": 1013.0,
                    "wind_speed": 15.0,
                    "cloud_cover": 30.0,
                    "month": 7.0,
                    "day_of_year": 200.0,
                }
            ]
        )
        result = predict(df)
        assert isinstance(result, dict)
        assert "predicted_temp" in result

    def test_predict_precip_non_negative(self):
        df = pd.DataFrame(
            [
                {
                    "temperature": -5.0,
                    "precipitation": 0.0,
                    "humidity": 90.0,
                    "pressure": 1005.0,
                    "wind_speed": 30.0,
                    "cloud_cover": 80.0,
                    "month": 1.0,
                    "day_of_year": 15.0,
                }
            ]
        )
        result = predict(df)
        assert result["predicted_precip"] >= 0

    def test_predict_extreme_prob_in_range(self):
        df = pd.DataFrame(
            [
                {
                    "temperature": 38.0,
                    "precipitation": 0.0,
                    "humidity": 20.0,
                    "pressure": 1020.0,
                    "wind_speed": 5.0,
                    "cloud_cover": 0.0,
                    "month": 7.0,
                    "day_of_year": 200.0,
                }
            ]
        )
        result = predict(df)
        p = result["extreme_event_prob"]
        assert 0.0 <= p <= 1.0


class TestGetMetrics:
    def test_returns_dict(self):
        m = get_metrics()
        assert isinstance(m, dict)

    def test_has_model_version(self):
        m = get_metrics()
        assert "model_version" in m
