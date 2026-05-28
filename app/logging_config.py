"""Structured logging configuration for Climate-Pulse.

Provides key=value structured log lines for easy parsing by log aggregators
such as Datadog, Loki, or CloudWatch Logs Insights.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

STRUCTURED_FIELDS = ("correlation_id", "station_id", "model_version", "request_id")


class StructuredFormatter(logging.Formatter):
    """Format log records as structured key=value lines.

    Extra fields attached to log records via ``extra=`` are appended
    in ``[key=value ...]`` notation after the base message.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record with appended structured context fields."""
        base = super().format(record)
        extra: dict[str, Any] = {}
        for key in STRUCTURED_FIELDS:
            if hasattr(record, key):
                extra[key] = getattr(record, key)
        if extra:
            kv = " ".join(f"{k}={v}" for k, v in extra.items())
            return f"{base} [{kv}]"
        return base


def configure_logging(level: str | None = None) -> None:
    """Configure root logger with structured formatting."""
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        StructuredFormatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
