"""Feature engineering pipeline tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    AtmosphericRatioTransformer,
    LagFeatureTransformer,
    RollingStatsTransformer,
    SeasonalEncodingTransformer,
    build_feature_pipeline,
    prepare_features,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "temperature": rng.normal(15, 8, 50),
            "precipitation": np.abs(rng.normal(2, 3, 50)),
            "humidity": rng.uniform(30, 95, 50),
            "pressure": rng.normal(1013, 10, 50),
            "wind_speed": np.abs(rng.normal(15, 8, 50)),
            "cloud_cover": rng.uniform(0, 100, 50),
            "month": rng.integers(1, 13, 50).astype(float),
            "day_of_year": rng.integers(1, 366, 50).astype(float),
        }
    )


class TestLagFeatureTransformer:
    def test_adds_lag_columns(self, sample_df):
        t = LagFeatureTransformer(lags=3)
        out = t.fit_transform(sample_df)
        assert "temp_lag_1" in out.columns
        assert "temp_lag_2" in out.columns
        assert "temp_lag_3" in out.columns

    def test_lag_1_is_shifted(self, sample_df):
        t = LagFeatureTransformer(lags=1)
        out = t.fit_transform(sample_df.reset_index(drop=True))
        assert out["temp_lag_1"].iloc[1] == pytest.approx(sample_df["temperature"].iloc[0], rel=1e-3)

    @pytest.mark.parametrize("lags", [1, 2, 5])
    def test_correct_number_of_lag_cols(self, sample_df, lags):
        t = LagFeatureTransformer(lags=lags)
        out = t.fit_transform(sample_df)
        lag_cols = [c for c in out.columns if c.startswith("temp_lag_")]
        assert len(lag_cols) == lags


class TestRollingStatsTransformer:
    def test_adds_rolling_mean(self, sample_df):
        t = RollingStatsTransformer(windows=[3])
        out = t.fit_transform(sample_df)
        assert "temp_roll_mean_3" in out.columns

    def test_adds_precip_rolling_sum(self, sample_df):
        t = RollingStatsTransformer(windows=[7])
        out = t.fit_transform(sample_df)
        assert "precip_roll_sum_7" in out.columns

    def test_no_nan_in_rolling(self, sample_df):
        t = RollingStatsTransformer(windows=[3, 7])
        out = t.fit_transform(sample_df)
        rolling_cols = [c for c in out.columns if "roll" in c]
        assert not out[rolling_cols].isna().any().any()


class TestAtmosphericRatioTransformer:
    def test_adds_humidity_pressure_ratio(self, sample_df):
        t = AtmosphericRatioTransformer()
        out = t.fit_transform(sample_df)
        assert "humidity_pressure_ratio" in out.columns

    def test_adds_wind_chill(self, sample_df):
        t = AtmosphericRatioTransformer()
        out = t.fit_transform(sample_df)
        assert "wind_chill" in out.columns

    def test_ratio_is_finite(self, sample_df):
        t = AtmosphericRatioTransformer()
        out = t.fit_transform(sample_df)
        assert np.isfinite(out["humidity_pressure_ratio"]).all()


class TestSeasonalEncodingTransformer:
    def test_adds_month_sin_cos(self, sample_df):
        t = SeasonalEncodingTransformer()
        out = t.fit_transform(sample_df)
        assert "month_sin" in out.columns
        assert "month_cos" in out.columns

    def test_sin_cos_in_range(self, sample_df):
        t = SeasonalEncodingTransformer()
        out = t.fit_transform(sample_df)
        assert out["month_sin"].between(-1, 1).all()
        assert out["month_cos"].between(-1, 1).all()


class TestBuildFeaturePipeline:
    def test_pipeline_transforms_without_error(self, sample_df):
        pipe = build_feature_pipeline()
        result = pipe.fit_transform(sample_df)
        assert result.shape[0] == len(sample_df)

    def test_output_has_more_features_than_input(self, sample_df):
        pipe = build_feature_pipeline()
        result = pipe.fit_transform(sample_df)
        assert result.shape[1] > sample_df.shape[1]

    def test_output_is_finite(self, sample_df):
        pipe = build_feature_pipeline()
        result = pipe.fit_transform(sample_df)
        assert np.isfinite(result).all()


class TestPrepareFeatures:
    def test_returns_dataframe(self):
        payload = {
            "temperature": 20.0, "precipitation": 2.0,
            "humidity": 60.0, "pressure": 1013.0,
            "wind_speed": 15.0, "cloud_cover": 30.0,
            "month": 6.0, "day_of_year": 160.0,
        }
        df = prepare_features(payload)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_missing_field_defaults_zero(self):
        df = prepare_features({"temperature": 20.0})
        assert df["precipitation"].iloc[0] == 0.0

    @pytest.mark.parametrize("feature", [
        "temperature", "precipitation", "humidity", "pressure",
        "wind_speed", "cloud_cover", "month", "day_of_year",
    ])
    def test_all_feature_columns_present(self, feature):
        df = prepare_features({})
        assert feature in df.columns

    def test_single_row_output(self):
        df = prepare_features({"temperature": 25.0, "humidity": 70.0})
        assert df.shape[0] == 1


class TestLagFeatureTransformerEdgeCases:
    def test_single_row_no_nan(self):
        single = pd.DataFrame({"temperature": [20.0], "precipitation": [2.0]})
        t = LagFeatureTransformer(lags=2)
        out = t.fit_transform(single)
        assert not out.isna().any().any()

    def test_precip_lag_zeros_for_nan(self, sample_df):
        t = LagFeatureTransformer(lags=1)
        out = t.fit_transform(sample_df)
        assert "precip_lag_1" in out.columns
        assert out["precip_lag_1"].iloc[0] == 0.0


class TestRollingStatsEdgeCases:
    @pytest.mark.parametrize("window", [3, 7, 14])
    def test_multiple_windows(self, sample_df, window):
        t = RollingStatsTransformer(windows=[window])
        out = t.fit_transform(sample_df)
        assert f"temp_roll_mean_{window}" in out.columns

    def test_default_windows_all_created(self, sample_df):
        t = RollingStatsTransformer()
        out = t.fit_transform(sample_df)
        for w in [3, 7, 14]:
            assert f"temp_roll_mean_{w}" in out.columns


class TestSeasonalEncodingEdgeCases:
    @pytest.mark.parametrize("month", [1.0, 6.0, 12.0])
    def test_specific_months_encoded(self, month):
        df = pd.DataFrame({"month": [month], "day_of_year": [180.0]})
        t = SeasonalEncodingTransformer()
        out = t.fit_transform(df)
        assert -1 <= out["month_sin"].iloc[0] <= 1

    def test_day_of_year_encoded(self, sample_df):
        t = SeasonalEncodingTransformer()
        out = t.fit_transform(sample_df)
        assert "doy_sin" in out.columns
        assert "doy_cos" in out.columns
