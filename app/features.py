"""Feature engineering pipeline for climate/weather prediction."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Adds lag-1 through lag-N columns for temperature and precipitation.

    Attributes:
        lags: Number of lag steps to generate (default 3).
    """

    def __init__(self, lags: int = 3) -> None:
        self.lags = lags

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> LagFeatureTransformer:  # noqa: N803
        """No-op fit — lag transformer is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Add lag columns to the input DataFrame.

        Args:
            X: Input feature DataFrame.

        Returns:
            DataFrame with added lag columns for temperature and precipitation.
        """
        X = X.copy()
        for lag in range(1, self.lags + 1):
            if "temperature" in X.columns:
                X[f"temp_lag_{lag}"] = X["temperature"].shift(lag).fillna(X["temperature"].mean())
            if "precipitation" in X.columns:
                X[f"precip_lag_{lag}"] = X["precipitation"].shift(lag).fillna(0.0)
        return X


class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Adds rolling mean/std over configurable timestep windows.

    Attributes:
        windows: List of window sizes in timesteps (default [3, 7, 14]).
    """

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [3, 7, 14]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RollingStatsTransformer:  # noqa: N803
        """No-op fit — rolling stats transformer is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        for w in self.windows:
            if "temperature" in X.columns:
                X[f"temp_roll_mean_{w}"] = X["temperature"].rolling(w, min_periods=1).mean()
                X[f"temp_roll_std_{w}"] = X["temperature"].rolling(w, min_periods=1).std().fillna(0)
            if "precipitation" in X.columns:
                X[f"precip_roll_sum_{w}"] = X["precipitation"].rolling(w, min_periods=1).sum()
        return X


class AtmosphericRatioTransformer(BaseEstimator, TransformerMixin):
    """Derives humidity-pressure ratio and wind chill index from raw observations."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> AtmosphericRatioTransformer:  # noqa: N803
        """No-op fit — ratio transformer is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        if "humidity" in X.columns and "pressure" in X.columns:
            X["humidity_pressure_ratio"] = X["humidity"] / (X["pressure"].replace(0, np.nan).fillna(1013.25))
        if "wind_speed" in X.columns and "temperature" in X.columns:
            X["wind_chill"] = (
                13.12
                + 0.6215 * X["temperature"]
                - 11.37 * (X["wind_speed"] ** 0.16)
                + 0.3965 * X["temperature"] * (X["wind_speed"] ** 0.16)
            )
        return X


class SeasonalEncodingTransformer(BaseEstimator, TransformerMixin):
    """Encodes month and day-of-year as sine/cosine cyclical features.

    Cyclical encoding preserves the periodic nature of temporal features
    (e.g., December is adjacent to January, day 365 to day 1).
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SeasonalEncodingTransformer:  # noqa: N803
        """No-op fit — encoding is deterministic."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        if "month" in X.columns:
            X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
            X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        if "day_of_year" in X.columns:
            X["doy_sin"] = np.sin(2 * np.pi * X["day_of_year"] / 365)
            X["doy_cos"] = np.cos(2 * np.pi * X["day_of_year"] / 365)
        return X


class DewiPointTransformer(BaseEstimator, TransformerMixin):
    """Approximates dew point temperature using the Magnus formula.

    Requires temperature (°C) and relative humidity (%) columns.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DewiPointTransformer:  # noqa: N803
        """No-op fit — dew point formula is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        if "temperature" in X.columns and "humidity" in X.columns:
            # Magnus formula approximation
            a, b = 17.27, 237.7
            alpha = (a * X["temperature"]) / (b + X["temperature"]) + np.log(X["humidity"].clip(1, 100) / 100)
            X["dew_point"] = (b * alpha) / (a - alpha)
        return X


class HeatIndexTransformer(BaseEstimator, TransformerMixin):
    """Computes the heat index (apparent temperature) via the Rothfusz regression equation.

    Valid for temperature > 26 °C and relative humidity > 40 %. Applied universally
    here; consumers should interpret below-threshold values as approximate.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> HeatIndexTransformer:  # noqa: N803
        """No-op fit — heat index formula is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        if "temperature" in X.columns and "humidity" in X.columns:
            T = X["temperature"]
            RH = X["humidity"].clip(0, 100)
            X["heat_index"] = (
                -8.78469475556
                + 1.61139411 * T
                + 2.33854883889 * RH
                - 0.14611605 * T * RH
                - 0.012308094 * T**2
                - 0.016424828 * RH**2
                + 0.002211732 * T**2 * RH
                + 0.00072546 * T * RH**2
                - 0.000003582 * T**2 * RH**2
            )
        return X


@lru_cache(maxsize=1)
def build_feature_pipeline() -> Pipeline:
    """Return the full 6-stage feature engineering + scaling pipeline.

    Cached so repeated calls in training loops return the same object.
    """
    return Pipeline(
        [
            ("lag_features", LagFeatureTransformer(lags=3)),
            ("rolling_stats", RollingStatsTransformer(windows=[3, 7, 14])),
            ("atmospheric_ratios", AtmosphericRatioTransformer()),
            ("seasonal_encoding", SeasonalEncodingTransformer()),
            ("dew_point", DewiPointTransformer()),
            ("heat_index", HeatIndexTransformer()),
            ("scaler", StandardScaler()),
        ]
    )


FEATURE_COLUMNS = [
    "temperature",
    "precipitation",
    "humidity",
    "pressure",
    "wind_speed",
    "cloud_cover",
    "month",
    "day_of_year",
]


def prepare_features(data: dict) -> pd.DataFrame:
    """Convert a prediction request dict into a single-row DataFrame with all features."""
    row = {col: data.get(col, 0.0) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row])
    logger.debug("features.prepare_features: shape=%s", df.shape)
    return df


def get_feature_names() -> list[str]:
    """Return the ordered list of raw input feature column names.

    Returns:
        List of feature name strings as expected by the prediction pipeline.
    """
    return list(FEATURE_COLUMNS)


def get_pipeline_stage_names() -> list[str]:
    """Return the names of all stages in the feature pipeline, excluding the final scaler.

    Returns:
        List of stage name strings (e.g. ``["lag_features", "rolling_stats", ...]``).
    """
    pipeline = build_feature_pipeline()
    return [name for name, _ in pipeline.steps if name != "scaler"]
