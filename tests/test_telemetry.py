"""Tests for telemetry counters and timing."""

from __future__ import annotations

import time

import pytest

from app.telemetry import Timer, _counters, _histograms, get_counter, get_stats, increment, record_latency, reset


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


class TestReset:
    def test_reset_clears_counters(self):
        increment("x", 5)
        reset()
        assert get_counter("x") == 0

    def test_reset_clears_histograms(self):
        record_latency("lat", 42.0)
        reset()
        stats = get_stats()
        assert "lat_p50" not in stats

    def test_reset_idempotent_on_empty_state(self):
        reset()
        reset()
        assert get_stats()["counters"] == {}


class TestGetCounter:
    def test_absent_metric_returns_zero(self):
        assert get_counter("nonexistent_metric_xyz") == 0

    def test_returns_current_count(self):
        increment("gc_test", 7)
        assert get_counter("gc_test") == 7

    def test_returns_accumulated_count(self):
        increment("acc_test", 3)
        increment("acc_test", 4)
        assert get_counter("acc_test") == 7


class TestGetCounterNames:
    def test_returns_list(self):
        from app.telemetry import get_counter_names

        result = get_counter_names()
        assert isinstance(result, list)

    def test_includes_incremented_counter(self):
        from app.telemetry import get_counter_names

        increment("my_test_counter")
        assert "my_test_counter" in get_counter_names()

    def test_sorted_alphabetically(self):
        from app.telemetry import get_counter_names

        increment("zzz_counter")
        increment("aaa_counter")
        names = get_counter_names()
        assert names == sorted(names)


class TestSnapshot:
    def test_returns_dict(self):
        from app.telemetry import snapshot

        result = snapshot()
        assert isinstance(result, dict)

    def test_contains_counters_and_histograms(self):
        from app.telemetry import snapshot

        result = snapshot()
        assert "counters" in result
        assert "histograms" in result

    def test_snapshot_is_copy(self):
        from app.telemetry import snapshot

        increment("snap_counter")
        s1 = snapshot()
        increment("snap_counter")
        s2 = snapshot()
        assert s2["counters"]["snap_counter"] > s1["counters"]["snap_counter"]

    def test_histogram_samples_in_snapshot(self):
        from app.telemetry import snapshot

        record_latency("snap_metric", 42.0)
        result = snapshot()
        assert "snap_metric" in result["histograms"]
        assert 42.0 in result["histograms"]["snap_metric"]

    @pytest.mark.parametrize("metric", ["requests", "errors", "retrains"])
    def test_multiple_counters_in_snapshot(self, metric):
        from app.telemetry import snapshot

        increment(metric, 5)
        result = snapshot()
        assert result["counters"].get(metric, 0) >= 5


class TestResetCounters:
    def test_resets_counters_only(self):
        from app.telemetry import reset_counters

        increment("alpha", 5)
        record_latency("beta_lat", 10.0)
        reset_counters()
        assert get_counter("alpha") == 0
        # histogram should still be intact
        stats = get_stats()
        assert "beta_lat_p50" in stats

    def test_reset_counters_idempotent(self):
        from app.telemetry import reset_counters

        reset_counters()
        reset_counters()
        assert get_stats()["counters"] == {}

    def test_reset_counters_does_not_affect_histograms(self):
        from app.telemetry import reset_counters

        record_latency("my_lat", 50.0)
        reset_counters()
        assert "my_lat" in get_stats()


class TestIncrementBatch:
    def test_increments_multiple_counters(self):
        from app.telemetry import increment_batch

        increment_batch({"a": 3, "b": 7, "c": 1})
        assert get_counter("a") == 3
        assert get_counter("b") == 7
        assert get_counter("c") == 1

    def test_accumulates_with_existing_values(self):
        from app.telemetry import increment_batch

        increment("x", 5)
        increment_batch({"x": 3, "y": 2})
        assert get_counter("x") == 8
        assert get_counter("y") == 2

    def test_empty_dict_is_no_op(self):
        from app.telemetry import increment_batch

        increment("existing", 4)
        increment_batch({})
        assert get_counter("existing") == 4

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_batch_size_parametrized(self, n):
        from app.telemetry import increment_batch

        metrics = {f"metric_{i}": i + 1 for i in range(n)}
        increment_batch(metrics)
        for name, val in metrics.items():
            assert get_counter(name) == val
