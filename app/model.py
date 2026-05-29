"""ML model training, persistence, and prediction for Climate-Pulse."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

from app.features import build_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(exist_ok=True)

TEMP_MODEL_PATH = MODEL_DIR / "temp_model.joblib"
PRECIP_MODEL_PATH = MODEL_DIR / "precip_model.joblib"
EXTREME_MODEL_PATH = MODEL_DIR / "extreme_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
MODEL_VERSION = "1.0.0"

_bundle_cache: dict[str, dict] = {}


class _EnsembleRegressor:
    """Average-prediction ensemble of heterogeneous regressors.

    Averages predictions from all member estimators. Avoids sklearn's
    VotingRegressor so there is no dependency on the evolving sklearn tags API.

    Attributes:
        estimators: List of fitted or unfitted sklearn-compatible regressors.
    """

    def __init__(self, estimators: list) -> None:
        self.estimators = estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> _EnsembleRegressor:
        """Fit all member estimators on the same training data.

        Args:
            X: Training feature matrix.
            y: Regression target vector.

        Returns:
            Self (fitted ensemble).
        """
        for est in self.estimators:
            est.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by averaging member estimator outputs.

        Args:
            X: Feature matrix for prediction.

        Returns:
            Averaged prediction array.
        """
        preds = np.column_stack([est.predict(X) for est in self.estimators])
        return preds.mean(axis=1)

    def get_params(self, deep: bool = True) -> dict:  # noqa: FBT001
        return {"estimators": self.estimators}


class _EnsembleClassifier:
    """Soft-vote (probability-averaging) ensemble of heterogeneous classifiers.

    Attributes:
        estimators: List of sklearn-compatible classifiers with predict_proba support.
    """

    def __init__(self, estimators: list) -> None:
        self.estimators = estimators

    def fit(self, X: np.ndarray, y: np.ndarray) -> _EnsembleClassifier:
        """Fit all member classifiers on the same training data.

        Args:
            X: Training feature matrix.
            y: Binary class label vector.

        Returns:
            Self (fitted ensemble).
        """
        for est in self.estimators:
            est.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class labels (threshold 0.5 on averaged probabilities).

        Args:
            X: Feature matrix for prediction.

        Returns:
            Binary class label array.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return averaged class probabilities from all member classifiers.

        Args:
            X: Feature matrix for prediction.

        Returns:
            Probability matrix of shape (n_samples, 2).
        """
        probas = np.stack([est.predict_proba(X) for est in self.estimators], axis=0)
        return probas.mean(axis=0)

    def get_params(self, deep: bool = True) -> dict:  # noqa: FBT001
        return {"estimators": self.estimators}


def _build_temp_ensemble() -> _EnsembleRegressor:
    return _EnsembleRegressor(
        [
            LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, verbose=-1),
            XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, verbosity=0),
            RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        ]
    )


def _build_precip_ensemble() -> _EnsembleRegressor:
    return _EnsembleRegressor(
        [
            LGBMRegressor(n_estimators=80, max_depth=4, learning_rate=0.08, verbose=-1),
            XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.08, verbosity=0),
            RandomForestRegressor(n_estimators=80, max_depth=5, random_state=42),
        ]
    )


def _build_extreme_ensemble() -> _EnsembleClassifier:
    return _EnsembleClassifier(
        [
            LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, verbose=-1),
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, verbosity=0, eval_metric="logloss"),
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        ]
    )


def _generate_synthetic_training_data(n: int = 2000) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Generate synthetic weather observations for model training."""
    rng = np.random.default_rng(42)
    months = rng.integers(1, 13, n)
    days = rng.integers(1, 366, n)

    base_temp = 15 + 12 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 3, n)
    temp_next = base_temp + rng.normal(0.5, 1.5, n)

    precip = np.maximum(0, rng.normal(2, 4, n) + 3 * (months <= 4).astype(float))
    precip_next = np.maximum(0, precip + rng.normal(0, 2, n))

    extreme = ((base_temp > 33) | (base_temp < -8) | (precip > 18)).astype(int)

    df = pd.DataFrame(
        {
            "temperature": base_temp,
            "precipitation": precip,
            "humidity": rng.uniform(30, 95, n),
            "pressure": rng.normal(1013, 10, n),
            "wind_speed": np.abs(rng.normal(15, 8, n)),
            "cloud_cover": rng.uniform(0, 100, n),
            "month": months.astype(float),
            "day_of_year": days.astype(float),
        }
    )
    return (
        df,
        pd.Series(temp_next, name="temp_next"),
        pd.Series(precip_next, name="precip_next"),
        pd.Series(extreme, name="extreme"),
    )


def train_models(
    X: pd.DataFrame | None = None,
    y_temp: pd.Series | None = None,
    y_precip: pd.Series | None = None,
    y_extreme: pd.Series | None = None,
) -> dict[str, Any]:
    """Train all three ensemble models with 5-fold CV; persist to disk."""
    if X is None:
        X, y_temp, y_precip, y_extreme = _generate_synthetic_training_data()
        logger.info("model.train_models: using synthetic training data n=%d", len(X))

    feat_pipe = build_feature_pipeline()
    X_eng = feat_pipe.fit_transform(X)

    temp_ensemble = _build_temp_ensemble()
    precip_ensemble = _build_precip_ensemble()
    extreme_ensemble = _build_extreme_ensemble()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # CV on individual base models (most compatible approach)
    lgbm_r = LGBMRegressor(n_estimators=100, max_depth=5, verbose=-1)
    temp_r2 = cross_val_score(lgbm_r, X_eng, y_temp, cv=kf, scoring="r2")

    lgbm_r2 = LGBMRegressor(n_estimators=80, max_depth=4, verbose=-1)
    precip_r2 = cross_val_score(lgbm_r2, X_eng, y_precip, cv=kf, scoring="r2")

    lgbm_c = LGBMClassifier(n_estimators=100, max_depth=5, verbose=-1)
    extreme_auc = cross_val_score(lgbm_c, X_eng, y_extreme, cv=skf, scoring="roc_auc")

    temp_ensemble.fit(X_eng, y_temp.to_numpy())
    precip_ensemble.fit(X_eng, y_precip.to_numpy())
    extreme_ensemble.fit(X_eng, y_extreme.to_numpy())

    metrics = {
        "temp_r2_mean": round(float(temp_r2.mean()), 4),
        "temp_r2_std": round(float(temp_r2.std()), 4),
        "precip_r2_mean": round(float(precip_r2.mean()), 4),
        "precip_r2_std": round(float(precip_r2.std()), 4),
        "extreme_auc_mean": round(float(extreme_auc.mean()), 4),
        "extreme_auc_std": round(float(extreme_auc.std()), 4),
        "n_training_samples": len(X),
        "n_features": X_eng.shape[1],
        "model_version": MODEL_VERSION,
    }

    joblib.dump({"pipeline": feat_pipe, "model": temp_ensemble}, TEMP_MODEL_PATH)
    joblib.dump({"pipeline": feat_pipe, "model": precip_ensemble}, PRECIP_MODEL_PATH)
    joblib.dump({"pipeline": feat_pipe, "model": extreme_ensemble}, EXTREME_MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    reset_model_cache()
    logger.info("model.train_models: done metrics=%s", metrics)
    return metrics


def _load_bundle(path: Path) -> dict[str, Any]:
    """Load a persisted model bundle, using an in-process cache to avoid repeated disk reads.

    Args:
        path: Filesystem path to the joblib bundle.

    Returns:
        Dict with keys ``pipeline`` (sklearn Pipeline) and ``model`` (ensemble).
    """
    key = str(path)
    if key in _bundle_cache:
        return _bundle_cache[key]
    if not path.exists():
        logger.warning("model: %s not found — training now", path.name)
        train_models()
        _bundle_cache.clear()
    bundle = joblib.load(path)
    _bundle_cache[key] = bundle
    return bundle


def reset_model_cache() -> None:
    """Evict all in-process model bundle cache entries.

    Call after retraining so subsequent predictions load the new models.
    """
    _bundle_cache.clear()
    logger.info("model.reset_model_cache: bundle cache cleared")


def is_model_trained() -> bool:
    """Return True if all three model files exist on disk.

    Returns:
        Boolean indicating whether persisted model files are present.
    """
    return TEMP_MODEL_PATH.exists() and PRECIP_MODEL_PATH.exists() and EXTREME_MODEL_PATH.exists()


def _transform_features(features: pd.DataFrame) -> np.ndarray:
    """Load the temp model bundle and apply its feature pipeline.

    Args:
        features: Raw input DataFrame from the predict endpoint.

    Returns:
        Engineered feature array ready for model inference.
    """
    bundle = _load_bundle(TEMP_MODEL_PATH)
    return bundle["pipeline"].transform(features)


def predict(features: pd.DataFrame) -> dict[str, float]:
    """Return predicted temperature, precipitation, and extreme event probability."""
    X_eng = _transform_features(features)

    predicted_temp = float(_load_bundle(TEMP_MODEL_PATH)["model"].predict(X_eng)[0])
    predicted_precip = float(np.maximum(0, _load_bundle(PRECIP_MODEL_PATH)["model"].predict(X_eng)[0]))
    extreme_proba = float(_load_bundle(EXTREME_MODEL_PATH)["model"].predict_proba(X_eng)[0][1])

    logger.debug(
        "model.predict: temp=%.2f precip=%.2f extreme_prob=%.4f",
        predicted_temp,
        predicted_precip,
        extreme_proba,
    )
    return {
        "predicted_temp": round(predicted_temp, 2),
        "predicted_precip": round(predicted_precip, 3),
        "extreme_event_prob": round(extreme_proba, 4),
        "model_version": MODEL_VERSION,
    }


def get_metrics() -> dict[str, Any]:
    """Load persisted training metrics."""
    if not METRICS_PATH.exists():
        train_models()
    return json.loads(METRICS_PATH.read_text())
