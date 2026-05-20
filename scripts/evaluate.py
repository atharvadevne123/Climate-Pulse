"""Evaluate Climate-Pulse models on a held-out test set and print metrics."""
from __future__ import annotations

import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

from app.model import (
    EXTREME_MODEL_PATH,
    PRECIP_MODEL_PATH,
    TEMP_MODEL_PATH,
    _generate_synthetic_training_data,
    _load_bundle,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def evaluate() -> dict:
    """Run evaluation on 500 held-out synthetic samples."""
    X, y_temp, y_precip, y_extreme = _generate_synthetic_training_data(n=500)

    temp_bundle = _load_bundle(TEMP_MODEL_PATH)
    precip_bundle = _load_bundle(PRECIP_MODEL_PATH)
    extreme_bundle = _load_bundle(EXTREME_MODEL_PATH)

    # Re-transform with the trained pipeline
    X_eval = temp_bundle["pipeline"].transform(X)

    temp_preds = temp_bundle["model"].predict(X_eval)
    precip_preds = np.maximum(0, precip_bundle["model"].predict(X_eval))
    extreme_proba = extreme_bundle["model"].predict_proba(X_eval)[:, 1]

    metrics = {
        "temp_r2": round(r2_score(y_temp, temp_preds), 4),
        "temp_mae": round(mean_absolute_error(y_temp, temp_preds), 4),
        "precip_r2": round(r2_score(y_precip, precip_preds), 4),
        "precip_mae": round(mean_absolute_error(y_precip, precip_preds), 4),
        "extreme_auc": round(roc_auc_score(y_extreme, extreme_proba), 4),
        "n_samples": len(X),
    }

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    print("=== Climate-Pulse Evaluation ===")
    evaluate()
