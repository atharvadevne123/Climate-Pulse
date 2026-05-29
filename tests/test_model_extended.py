"""Extended model tests — synthetic data generation, ensemble, and predict edge cases."""

from __future__ import annotations

import pandas as pd
import pytest

from app.model import (
    _EnsembleClassifier,
    _EnsembleRegressor,
    _generate_synthetic_training_data,
    _transform_features,
    get_metrics,
    predict,
    train_models,
)


@pytest.fixture(scope="module")
def trained():
    return train_models()


class TestSyntheticDataEdgeCases:
    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_various_sample_sizes(self, n):
        X, y_temp, y_precip, y_extreme = _generate_synthetic_training_data(n=n)
        assert len(X) == n

    def test_feature_columns_complete(self):
        X, _, _, _ = _generate_synthetic_training_data(n=100)
        expected = {
            "temperature",
            "precipitation",
            "humidity",
            "pressure",
            "wind_speed",
            "cloud_cover",
            "month",
            "day_of_year",
        }
        assert expected.issubset(set(X.columns))

    def test_month_in_valid_range(self):
        X, _, _, _ = _generate_synthetic_training_data(n=500)
        assert X["month"].between(1, 12).all()

    def test_day_of_year_in_valid_range(self):
        X, _, _, _ = _generate_synthetic_training_data(n=500)
        assert X["day_of_year"].between(1, 365).all()

    def test_humidity_in_range(self):
        X, _, _, _ = _generate_synthetic_training_data(n=500)
        assert X["humidity"].between(30, 95).all()

    def test_deterministic_with_same_seed(self):
        X1, _, _, _ = _generate_synthetic_training_data(n=100)
        X2, _, _, _ = _generate_synthetic_training_data(n=100)
        assert (X1["temperature"].values == X2["temperature"].values).all()


class TestEnsembleRegressor:
    def test_fit_predict(self):
        import numpy as np
        from sklearn.linear_model import LinearRegression

        estimators = [LinearRegression(), LinearRegression()]
        ens = _EnsembleRegressor(estimators)
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        ens.fit(X, y)
        preds = ens.predict(X)
        assert preds.shape == (3,)

    def test_predict_average(self):
        import numpy as np
        from sklearn.linear_model import LinearRegression

        est1 = LinearRegression().fit(np.array([[1.0], [2.0]]), np.array([2.0, 4.0]))
        est2 = LinearRegression().fit(np.array([[1.0], [2.0]]), np.array([2.0, 4.0]))
        ens = _EnsembleRegressor([est1, est2])
        result = ens.predict(np.array([[3.0]]))
        assert abs(result[0] - 6.0) < 0.5

    def test_get_params(self):
        from sklearn.linear_model import LinearRegression

        ens = _EnsembleRegressor([LinearRegression()])
        params = ens.get_params()
        assert "estimators" in params


class TestEnsembleClassifier:
    def test_fit_predict_proba(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        estimators = [LogisticRegression(), LogisticRegression()]
        ens = _EnsembleClassifier(estimators)
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        y = np.array([1, 0, 1, 0])
        ens.fit(X, y)
        proba = ens.predict_proba(X)
        assert proba.shape == (4, 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        estimators = [LogisticRegression()]
        ens = _EnsembleClassifier(estimators)
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([1, 0])
        ens.fit(X, y)
        preds = ens.predict(X)
        assert set(preds).issubset({0, 1})


class TestPredictEdgeCases:
    def _make_input(self, **overrides) -> pd.DataFrame:
        base = {
            "temperature": 20.0,
            "precipitation": 2.0,
            "humidity": 60.0,
            "pressure": 1013.0,
            "wind_speed": 15.0,
            "cloud_cover": 30.0,
            "month": 6.0,
            "day_of_year": 160.0,
        }
        base.update(overrides)
        return pd.DataFrame([base])

    @pytest.mark.parametrize("temperature", [-5.0, 0.0, 20.0, 35.0])
    def test_predict_various_temperatures(self, temperature):
        result = predict(self._make_input(temperature=temperature))
        assert "predicted_temp" in result

    def test_predict_has_model_version(self):
        result = predict(self._make_input())
        assert "model_version" in result
        assert isinstance(result["model_version"], str)

    @pytest.mark.parametrize("precipitation", [0.0, 5.0, 50.0])
    def test_predict_precip_non_negative(self, precipitation):
        result = predict(self._make_input(precipitation=precipitation))
        assert result["predicted_precip"] >= 0.0


class TestGetMetricsEdgeCases:
    def test_all_required_keys_present(self):
        m = get_metrics()
        required = {
            "temp_r2_mean",
            "precip_r2_mean",
            "extreme_auc_mean",
            "n_training_samples",
            "n_features",
            "model_version",
        }
        assert required.issubset(m.keys())

    def test_n_training_samples_is_int(self):
        m = get_metrics()
        assert isinstance(m["n_training_samples"], int)

    def test_n_features_positive(self):
        m = get_metrics()
        assert m["n_features"] > 0


class TestTransformFeatures:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "temperature": 20.0,
                    "precipitation": 2.0,
                    "humidity": 60.0,
                    "pressure": 1013.0,
                    "wind_speed": 15.0,
                    "cloud_cover": 30.0,
                    "month": 6.0,
                    "day_of_year": 160.0,
                }
            ]
        )

    def test_returns_array(self):
        import numpy as np

        result = _transform_features(self._make_df())
        assert isinstance(result, np.ndarray)

    def test_output_row_count_matches_input(self):
        df = self._make_df()
        result = _transform_features(df)
        assert result.shape[0] == len(df)

    def test_output_has_more_features_than_input(self):
        df = self._make_df()
        result = _transform_features(df)
        assert result.shape[1] > df.shape[1]

    def test_output_is_finite(self):
        import numpy as np

        result = _transform_features(self._make_df())
        assert np.isfinite(result).all()


class TestPredictEdgeCases:
    def _payload(self, **overrides):
        base = {
            "temperature": 20.0, "precipitation": 2.0, "humidity": 60.0,
            "pressure": 1013.0, "wind_speed": 15.0, "cloud_cover": 30.0,
            "month": 6.0, "day_of_year": 160.0,
        }
        base.update(overrides)
        return pd.DataFrame([base])

    @pytest.mark.parametrize("temperature", [-89.0, 0.0, 35.0, 59.0])
    def test_predict_valid_temperature_range(self, temperature):
        result = predict(self._payload(temperature=temperature))
        assert "predicted_temp" in result

    def test_predict_extreme_temperature_hot(self):
        result = predict(self._payload(temperature=55.0, humidity=90.0))
        assert "extreme_event_prob" in result
        assert 0.0 <= result["extreme_event_prob"] <= 1.0

    def test_predict_extreme_temperature_cold(self):
        result = predict(self._payload(temperature=-30.0))
        assert "extreme_event_prob" in result

    def test_predict_zero_wind(self):
        result = predict(self._payload(wind_speed=0.0))
        assert "predicted_temp" in result

    def test_predict_full_cloud_cover(self):
        result = predict(self._payload(cloud_cover=100.0))
        assert "predicted_temp" in result

    @pytest.mark.parametrize("month", [1.0, 6.0, 12.0])
    def test_predict_seasonal_months(self, month):
        result = predict(self._payload(month=month))
        assert "predicted_temp" in result

    def test_predict_model_version_string(self):
        result = predict(self._payload())
        assert isinstance(result["model_version"], str)
        assert len(result["model_version"]) > 0

    def test_predict_precip_non_negative(self):
        for _ in range(5):
            result = predict(self._payload())
            assert result["predicted_precip"] >= 0.0
