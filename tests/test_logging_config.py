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


class TestStructuredFormatterParametrized:
    def test_all_structured_fields_recognized(self):
        from app.logging_config import STRUCTURED_FIELDS, StructuredFormatter
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        for field in STRUCTURED_FIELDS:
            setattr(record, field, f"{field}_value")
        result = fmt.format(record)
        for field in STRUCTURED_FIELDS:
            assert f"{field}_value" in result

    def test_format_with_all_log_levels(self):
        import pytest

        from app.logging_config import StructuredFormatter

        @pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR])
        def inner(level):
            fmt = StructuredFormatter()
            record = logging.LogRecord("test", level, "", 0, "msg", (), None)
            result = fmt.format(record)
            assert "msg" in result

        for level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
            fmt = StructuredFormatter()
            record = logging.LogRecord("test", level, "", 0, "msg", (), None)
            result = fmt.format(record)
            assert "msg" in result

    def test_configure_logging_accepts_lowercase(self):
        from app.logging_config import configure_logging
        configure_logging("info")
        configure_logging("INFO")  # restore to INFO

    def test_configure_logging_debug_level(self):
        from app.logging_config import configure_logging
        configure_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        configure_logging("INFO")  # restore


class TestStructuredFormatterEdgeCases:
    def test_empty_message_still_formats(self):
        from app.logging_config import StructuredFormatter
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_none_correlation_id_handled(self):
        from app.logging_config import StructuredFormatter
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        record.correlation_id = None
        result = fmt.format(record)
        assert isinstance(result, str)

    def test_formatter_is_deterministic(self):
        from app.logging_config import StructuredFormatter
        fmt = StructuredFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "deterministic", (), None)
        r1 = fmt.format(record)
        r2 = fmt.format(record)
        assert r1 == r2
