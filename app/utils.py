"""General-purpose utility functions for Climate-Pulse."""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


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
