"""Initial schema — prediction_logs, drift_reports, weather_observations.

Revision ID: 001
Revises:
Create Date: 2026-05-20
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("correlation_id", sa.String, index=True),
        sa.Column("timestamp", sa.DateTime),
        sa.Column("station_id", sa.String, index=True),
        sa.Column("features", sa.JSON),
        sa.Column("predicted_temp", sa.Float),
        sa.Column("predicted_precip", sa.Float),
        sa.Column("extreme_event_prob", sa.Float),
        sa.Column("model_version", sa.String),
    )
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("timestamp", sa.DateTime),
        sa.Column("feature_name", sa.String, index=True),
        sa.Column("ks_statistic", sa.Float),
        sa.Column("p_value", sa.Float),
        sa.Column("drift_detected", sa.Integer),
    )
    op.create_table(
        "weather_observations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("station_id", sa.String, index=True),
        sa.Column("timestamp", sa.DateTime, index=True),
        sa.Column("temperature", sa.Float),
        sa.Column("precipitation", sa.Float),
        sa.Column("humidity", sa.Float),
        sa.Column("pressure", sa.Float),
        sa.Column("wind_speed", sa.Float),
        sa.Column("cloud_cover", sa.Float),
    )


def downgrade() -> None:
    op.drop_table("weather_observations")
    op.drop_table("drift_reports")
    op.drop_table("prediction_logs")
