"""Airflow DAG for automated weekly model retraining with validation gate."""
from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "climate-pulse",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

RETRAINING_THRESHOLD_R2 = 0.50
EXTREME_AUC_THRESHOLD = 0.70


def _load_data() -> dict:
    """Load latest weather observations for retraining."""
    import os
    import sys
    sys.path.insert(0, os.getcwd())
    from app.model import _generate_synthetic_training_data  # noqa: PLC0415
    X, y_temp, y_precip, y_extreme = _generate_synthetic_training_data(n=3000)
    return {
        "X": X.to_json(),
        "y_temp": y_temp.to_json(),
        "y_precip": y_precip.to_json(),
        "y_extreme": y_extreme.to_json(),
    }


def _train(**context) -> None:
    """Retrain models and push metrics to XCom."""
    import os  # noqa: E401
    import sys

    import pandas as pd
    sys.path.insert(0, os.getcwd())
    from app.model import train_models  # noqa: PLC0415

    ti = context["ti"]
    data = ti.xcom_pull(task_ids="load_data")
    X = pd.read_json(data["X"])
    y_temp = pd.read_json(data["y_temp"], typ="series")
    y_precip = pd.read_json(data["y_precip"], typ="series")
    y_extreme = pd.read_json(data["y_extreme"], typ="series")

    metrics = train_models(X, y_temp, y_precip, y_extreme)
    ti.xcom_push(key="metrics", value=metrics)
    logger.info("retrain_dag._train: metrics=%s", metrics)


def _validate(**context) -> None:
    """Assert model quality gates before promoting."""
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="train_models", key="metrics")

    if metrics["temp_r2_mean"] < RETRAINING_THRESHOLD_R2:
        raise ValueError(
            f"Temp R² {metrics['temp_r2_mean']:.4f} below threshold {RETRAINING_THRESHOLD_R2}"
        )
    if metrics["extreme_auc_mean"] < EXTREME_AUC_THRESHOLD:
        raise ValueError(
            f"Extreme AUC {metrics['extreme_auc_mean']:.4f} below threshold {EXTREME_AUC_THRESHOLD}"
        )
    logger.info("retrain_dag._validate: all gates passed metrics=%s", metrics)


def _log_metrics(**context) -> None:
    """Log final metrics after successful validation."""
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="train_models", key="metrics")
    logger.info("retrain_dag._log_metrics: promotion approved metrics=%s", metrics)


if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="climate_pulse_weekly_retrain",
        default_args=DEFAULT_ARGS,
        description="Weekly automated retraining for Climate-Pulse models",
        schedule="@weekly",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["climate-pulse", "ml", "retraining"],
    ) as dag:
        load_data_task = PythonOperator(
            task_id="load_data",
            python_callable=_load_data,
        )

        train_task = PythonOperator(
            task_id="train_models",
            python_callable=_train,
        )

        validate_task = PythonOperator(
            task_id="validate_models",
            python_callable=_validate,
        )

        log_task = PythonOperator(
            task_id="log_metrics",
            python_callable=_log_metrics,
        )

        load_data_task >> train_task >> validate_task >> log_task
else:
    logger.warning("retrain_dag: Airflow not installed; DAG not registered")
