"""General-purpose utility functions for Climate-Pulse."""

from __future__ import annotations

import logging
import math
from collections.abc import Generator, Iterable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def clamp(value: float, lo: float, hi: float) -> float:
    """Return value clamped to the closed interval [lo, hi].

    Args:
        value: The value to clamp.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        Clamped float in [lo, hi].
    """
    return max(lo, min(hi, value))


def round_to_sig_figs(value: float, sig_figs: int = 4) -> float:
    """Round a float to the given number of significant figures.

    Args:
        value: Number to round.
        sig_figs: Significant figures to retain (default 4).

    Returns:
        Rounded float value.
    """
    if value == 0.0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    factor = 10 ** (sig_figs - 1 - magnitude)
    return round(value * factor) / factor


def flatten_dict(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Recursively flatten a nested dictionary into dot-separated keys.

    Args:
        d: Nested dictionary to flatten.
        prefix: Key prefix to prepend (used internally for recursion).
        sep: Separator between key segments (default ".").

    Returns:
        Flat dict with compound keys like ``"a.b.c"``.
    """
    result: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, full_key, sep))
        else:
            result[full_key] = value
    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce *value* to float, returning *default* if conversion fails.

    Args:
        value: Any value to coerce.
        default: Fallback value on failure (default 0.0).

    Returns:
        Float representation of *value*, or *default*.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("utils.safe_float: could not convert %r to float", value)
        return default


def percentage_change(old: float, new: float) -> float:
    """Return the percentage change from *old* to *new*.

    Args:
        old: Reference value (denominator).
        new: New value (numerator target).

    Returns:
        Percentage change as a float (e.g. 10.0 means +10 %). Returns 0.0
        when *old* is zero to avoid division errors.
    """
    if old == 0.0:
        logger.debug("utils.percentage_change: old=0, returning 0.0")
        return 0.0
    return (new - old) / abs(old) * 100.0


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds (non-negative).

    Returns:
        String such as ``"2h 3m 5s"`` or ``"45.3s"`` for sub-minute durations.
    """
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.0f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.0f}s"
    return f"{secs:.1f}s"


def moving_average(values: list[float], window: int) -> list[float]:
    """Return a list of moving averages over *window* elements.

    Args:
        values: Input sequence of floats.
        window: Number of preceding values to average (must be >= 1).

    Returns:
        List of the same length as *values*; early values use a shrinking window.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    result: list[float] = []
    for i, _ in enumerate(values):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, falling back to *default* when denominator is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero (default 0.0).

    Returns:
        Division result or *default*.
    """
    if denominator == 0.0:
        logger.debug("utils.safe_divide: denominator=0, returning default=%s", default)
        return default
    return numerator / denominator


def batch_iter(iterable: Iterable[T], batch_size: int) -> Generator[list[T], None, None]:
    """Yield successive fixed-size batches from *iterable*.

    Args:
        iterable: Any iterable to chunk.
        batch_size: Maximum number of items per batch (must be >= 1).

    Yields:
        Lists of up to *batch_size* items.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
