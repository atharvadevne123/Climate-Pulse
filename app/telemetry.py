"""Lightweight telemetry counters and timing utilities."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

_counters: dict[str, int] = defaultdict(int)
_histograms: dict[str, list[float]] = defaultdict(list)


def increment(metric: str, value: int = 1) -> None:
    """Increment a named counter."""
    _counters[metric] += value


def record_latency(metric: str, duration_ms: float) -> None:
    """Append a latency sample to a histogram."""
    _histograms[metric].append(duration_ms)
    if len(_histograms[metric]) > 1000:
        _histograms[metric] = _histograms[metric][-500:]


def get_stats() -> dict[str, Any]:
    """Return current counters and histogram p50/p95/p99."""
    stats: dict[str, Any] = {"counters": dict(_counters)}
    for name, samples in _histograms.items():
        if not samples:
            continue
        sorted_s = sorted(samples)
        n = len(sorted_s)
        stats[f"{name}_p50"] = round(sorted_s[int(n * 0.5)], 2)
        stats[f"{name}_p95"] = round(sorted_s[int(n * 0.95)], 2)
        stats[f"{name}_p99"] = round(sorted_s[min(int(n * 0.99), n - 1)], 2)
        stats[f"{name}_count"] = n
    return stats


class Timer:
    """Context manager that records elapsed time as a latency sample."""

    def __init__(self, metric: str) -> None:
        self.metric = metric
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        record_latency(self.metric, elapsed_ms)
        logger.debug("telemetry.Timer: %s=%.2fms", self.metric, elapsed_ms)
