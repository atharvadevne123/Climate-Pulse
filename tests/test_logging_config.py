"""Tests for structured logging configuration."""
from __future__ import annotations

import logging

from app.logging_config import StructuredFormatter, configure_logging


class TestStructuredFormatter:
    def test_formats_basic_message(self):
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        result = fmt.format(record)
        assert "hello" in result

    def test_includes_extra_fields(self):
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.correlation_id = "corr-abc"
        result = fmt.format(record)
        assert "corr-abc" in result


class TestConfigureLogging:
    def test_configure_sets_level(self):
        configure_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING
        configure_logging("INFO")

    def test_configure_no_error(self):
        configure_logging()
