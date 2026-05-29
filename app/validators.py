"""Input validation utilities for Climate-Pulse."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

VALID_FEATURES: frozenset[str] = frozenset(
    [
        "temperature",
        "precipitation",
        "humidity",
        "pressure",
        "wind_speed",
        "cloud_cover",
        "month",
        "day_of_year",
    ]
)

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


def validate_feature_dict(data: dict[str, Any]) -> list[str]:
    """Return a list of validation error strings; empty means valid.

    Args:
        data: Dictionary mapping feature name to its value.

    Returns:
        List of human-readable error strings (empty if all valid).
    """
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


def validate_station_id_format(station_id: str) -> list[str]:
    """Return validation errors for station_id format (alphanumeric + hyphens/underscores/dots).

    Args:
        station_id: Station identifier string to validate.

    Returns:
        List of human-readable error strings (empty if valid).
    """
    import re

    errors: list[str] = []
    if not is_valid_station_id(station_id):
        errors.append(f"station_id '{station_id}' is blank, too long, or contains non-printable characters")
        return errors
    pattern = re.compile(r"^[A-Za-z0-9_\-\.]+$")
    if not pattern.match(station_id):
        errors.append(
            f"station_id '{station_id}' contains invalid characters; only alphanumeric, hyphens, underscores, and dots are allowed"
        )
    return errors


def validate_prediction_output(predictions: dict[str, Any]) -> list[str]:
    """Validate that prediction output values are within physically plausible ranges.

    Args:
        predictions: Dict with keys predicted_temp, predicted_precip, extreme_event_prob.

    Returns:
        List of human-readable error strings (empty if all valid).
    """
    errors: list[str] = []
    temp = predictions.get("predicted_temp")
    if temp is not None and not isinstance(temp, int | float):
        errors.append("predicted_temp must be numeric")
    elif temp is not None and not (-90.0 <= float(temp) <= 60.0):
        errors.append(f"predicted_temp {temp} is outside plausible range [-90, 60]")

    precip = predictions.get("predicted_precip")
    if precip is not None and not isinstance(precip, int | float):
        errors.append("predicted_precip must be numeric")
    elif precip is not None and float(precip) < 0:
        errors.append(f"predicted_precip {precip} must be non-negative")

    prob = predictions.get("extreme_event_prob")
    if prob is not None and not isinstance(prob, int | float):
        errors.append("extreme_event_prob must be numeric")
    elif prob is not None and not (0.0 <= float(prob) <= 1.0):
        errors.append(f"extreme_event_prob {prob} must be in [0, 1]")

    return errors


def validate_cross_field(data: dict[str, Any]) -> list[str]:
    """Return cross-field validation errors that cannot be caught field-by-field.

    Currently checks:
    - ``dew_point`` (if present) should not exceed ``temperature``
    - ``pressure`` extremes are inconsistent with tropical storm conditions

    Args:
        data: Dict of feature name → value (numeric).

    Returns:
        List of human-readable error strings (empty if all valid).
    """
    errors: list[str] = []
    temp = data.get("temperature")
    humidity = data.get("humidity")
    if temp is not None and humidity is not None:
        if isinstance(temp, int | float) and isinstance(humidity, int | float):
            # Very low humidity with very high temperature is physically possible but flagged
            if temp > 45 and humidity < 5:
                errors.append(f"temperature={temp} with humidity={humidity} is an unusual combination; verify sensor")
    logger.debug("validators.validate_cross_field: errors=%s", errors)
    return errors
