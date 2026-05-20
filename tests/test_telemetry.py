"""Tests for telemetry counters and timing."""
from __future__ import annotations

import time

import pytest

from app.telemetry import Timer, _counters, _histograms, get_stats, increment, record_latency


@pytest.fixture(autouse=True)
def reset_telemetry():
    _counters.clear()
    _histograms.clear()
    yield
    _counters.clear()
    _histograms.clear()


class TestIncrement:
    def test_increment_default(self):
        increment("requests")
        assert get_stats()["counters"]["requests"] == 1

    def test_increment_multiple(self):
        increment("requests", 3)
        increment("requests", 2)
        assert get_stats()["counters"]["requests"] == 5

    def test_multiple_counters(self):
        increment("a")
        increment("b", 5)
        stats = get_stats()
        assert stats["counters"]["a"] == 1
        assert stats["counters"]["b"] == 5


class TestRecordLatency:
    def test_records_sample(self):
        record_latency("predict_ms", 42.0)
        stats = get_stats()
        assert "predict_ms_p50" in stats
        assert stats["predict_ms_count"] == 1

    def test_percentiles_reasonable(self):
        for i in range(100):
            record_latency("req", float(i))
        stats = get_stats()
        assert stats["req_p50"] < stats["req_p95"] < stats["req_p99"]


class TestTimer:
    def test_timer_records_latency(self):
        with Timer("test_op"):
            time.sleep(0.01)
        stats = get_stats()
        assert "test_op_p50" in stats
        assert stats["test_op_p50"] >= 10.0
