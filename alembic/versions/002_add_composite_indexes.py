"""Add composite indexes for common query patterns.

Revision ID: 002
Revises: 001
Create Date: 2026-05-29
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Composite index to speed up station+timestamp range queries
    op.create_index(
        "ix_prediction_logs_station_timestamp",
        "prediction_logs",
        ["station_id", "timestamp"],
    )
    # Index for model_version filtering in analytics queries
    op.create_index(
        "ix_prediction_logs_model_version",
        "prediction_logs",
        ["model_version"],
    )
    # Composite index to speed up per-feature drift history lookups
    op.create_index(
        "ix_drift_reports_feature_timestamp",
        "drift_reports",
        ["feature_name", "timestamp"],
    )
    # Index to quickly count drift events (drift_detected = 1)
    op.create_index(
        "ix_drift_reports_drift_detected",
        "drift_reports",
        ["drift_detected"],
    )


def downgrade() -> None:
    op.drop_index("ix_drift_reports_drift_detected", table_name="drift_reports")
    op.drop_index("ix_drift_reports_feature_timestamp", table_name="drift_reports")
    op.drop_index("ix_prediction_logs_model_version", table_name="prediction_logs")
    op.drop_index("ix_prediction_logs_station_timestamp", table_name="prediction_logs")
