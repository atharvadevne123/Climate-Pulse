"""Input validation utilities for Climate-Pulse."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VALID_FEATURES = frozenset([
    "temperature", "precipitation", "humidity", "pressure",
    "wind_speed", "cloud_cover", "month", "day_of_year",
])

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "temperature": (-90.0, 60.0),
    "precipitation": (0.0, 500.0),
    "humidity": (0.0, 100.0),
    "pressure": (870.0, 1085.0),
    "wind_speed": (0.0, 200.0),
    "cloud_cover": (0.0, 100.0),
    "month": (1.0, 12.0),
    "day_of_year": (1.0, 366.0),
}


def validate_feature_dict(data: dict) -> list[str]:
    """Return a list of validation error strings; empty means valid."""
    errors: list[str] = []
    for feature, (lo, hi) in FEATURE_RANGES.items():
        if feature not in data:
            continue
        val = data[feature]
        if not isinstance(val, int | float):
            errors.append(f"{feature}: expected numeric, got {type(val).__name__}")
        elif not (lo <= val <= hi):
            errors.append(f"{feature}: {val} out of range [{lo}, {hi}]")
    missing = [f for f in FEATURE_RANGES if f not in data]
    if missing:
        errors.append(f"missing required features: {missing}")
    logger.debug("validators.validate_feature_dict: errors=%s", errors)
    return errors


STATION_ID_MAX_LEN = 64


def is_valid_station_id(station_id: str) -> bool:
    """Return True if station_id is a non-empty, printable ASCII string within length limits."""
    if not station_id or not station_id.strip():
        return False
    if len(station_id) > STATION_ID_MAX_LEN:
        logger.warning("validators.is_valid_station_id: station_id too long len=%d", len(station_id))
        return False
    return station_id.isprintable()
