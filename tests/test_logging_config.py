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


class TestStructuredFormatterThreadId:
    def test_thread_id_in_output(self):
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = fmt.format(record)
        assert "thread=" in result

    def test_structured_fields_and_thread_combined(self):
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.station_id = "STATION_42"
        result = fmt.format(record)
        assert "station_id=STATION_42" in result
        assert "thread=" in result

    def test_no_extra_fields_still_has_thread(self):
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.DEBUG, "", 0, "debug msg", (), None)
        result = fmt.format(record)
        assert "thread=" in result

    def test_model_version_in_output(self):
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.model_version = "2.0.0"
        result = fmt.format(record)
        assert "model_version=2.0.0" in result
