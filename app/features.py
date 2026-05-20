"""Feature engineering pipeline for climate/weather prediction."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Adds lag-1, lag-2, lag-3 columns for temperature and precipitation."""

    def __init__(self, lags: int = 3) -> None:
        self.lags = lags

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        for lag in range(1, self.lags + 1):
            if "temperature" in X.columns:
                X[f"temp_lag_{lag}"] = X["temperature"].shift(lag).fillna(X["temperature"].mean())
            if "precipitation" in X.columns:
                X[f"precip_lag_{lag}"] = X["precipitation"].shift(lag).fillna(0.0)
        return X


class RollingStatsTransformer(BaseEstimator, TransformerMixin):
    """Adds rolling mean/std over 3, 7, and 14 timestep windows."""

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or [3, 7, 14]

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        for w in self.windows:
            if "temperature" in X.columns:
                X[f"temp_roll_mean_{w}"] = (
                    X["temperature"].rolling(w, min_periods=1).mean()
                )
                X[f"temp_roll_std_{w}"] = (
                    X["temperature"].rolling(w, min_periods=1).std().fillna(0)
                )
            if "precipitation" in X.columns:
                X[f"precip_roll_sum_{w}"] = (
                    X["precipitation"].rolling(w, min_periods=1).sum()
                )
        return X


class AtmosphericRatioTransformer(BaseEstimator, TransformerMixin):
    """Derives humidity-pressure ratio and wind chill index."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
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
    """Encodes month/day-of-year using sine/cosine cyclical encoding."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
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
    """Approximates dew point temperature from temperature and relative humidity."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = X.copy()
        if "temperature" in X.columns and "humidity" in X.columns:
            # Magnus formula approximation
            a, b = 17.27, 237.7
            alpha = (a * X["temperature"]) / (b + X["temperature"]) + np.log(
                X["humidity"].clip(1, 100) / 100
            )
            X["dew_point"] = (b * alpha) / (a - alpha)
        return X


def build_feature_pipeline() -> Pipeline:
    """Return the full 5-stage feature engineering + scaling pipeline."""
    return Pipeline(
        [
            ("lag_features", LagFeatureTransformer(lags=3)),
            ("rolling_stats", RollingStatsTransformer(windows=[3, 7, 14])),
            ("atmospheric_ratios", AtmosphericRatioTransformer()),
            ("seasonal_encoding", SeasonalEncodingTransformer()),
            ("dew_point", DewiPointTransformer()),
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
