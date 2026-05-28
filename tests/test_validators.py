"""Tests for input validation utilities."""
from __future__ import annotations

import pytest

from app.validators import is_valid_station_id, validate_feature_dict

VALID_PAYLOAD = {
    "temperature": 20.0,
    "precipitation": 2.0,
    "humidity": 60.0,
    "pressure": 1013.0,
    "wind_speed": 15.0,
    "cloud_cover": 30.0,
    "month": 6.0,
    "day_of_year": 160.0,
}


class TestValidateFeatureDict:
    def test_valid_payload_no_errors(self):
        assert validate_feature_dict(VALID_PAYLOAD) == []

    @pytest.mark.parametrize("field,bad_value", [
        ("temperature", 200.0),
        ("humidity", -5.0),
        ("pressure", 500.0),
        ("month", 13.0),
        ("day_of_year", 400.0),
    ])
    def test_out_of_range_returns_error(self, field, bad_value):
        payload = {**VALID_PAYLOAD, field: bad_value}
        errors = validate_feature_dict(payload)
        assert any(field in e for e in errors)

    def test_missing_features_flagged(self):
        errors = validate_feature_dict({})
        assert any("missing" in e for e in errors)

    def test_non_numeric_value_flagged(self):
        payload = {**VALID_PAYLOAD, "temperature": "hot"}
        errors = validate_feature_dict(payload)
        assert any("temperature" in e for e in errors)


class TestIsValidStationId:
    def test_valid_station_id(self):
        assert is_valid_station_id("STATION_001") is True

    @pytest.mark.parametrize("bad", ["", "   ", "\x00invalid"])
    def test_invalid_station_ids(self, bad):
        assert is_valid_station_id(bad) is False

    def test_station_id_at_max_length_is_valid(self):
        from app.validators import STATION_ID_MAX_LEN
        assert is_valid_station_id("A" * STATION_ID_MAX_LEN) is True

    def test_station_id_exceeds_max_length_is_invalid(self):
        from app.validators import STATION_ID_MAX_LEN
        assert is_valid_station_id("A" * (STATION_ID_MAX_LEN + 1)) is False

    @pytest.mark.parametrize("valid_id", ["AWS-001", "station.42", "NYC_JFK_INTL", "S1"])
    def test_valid_station_id_formats(self, valid_id):
        assert is_valid_station_id(valid_id) is True


class TestValidateFeatureDictEdgeCases:
    @pytest.mark.parametrize("field,boundary_value", [
        ("temperature", -90.0),
        ("temperature", 60.0),
        ("precipitation", 0.0),
        ("humidity", 0.0),
        ("humidity", 100.0),
        ("month", 1.0),
        ("month", 12.0),
    ])
    def test_boundary_values_are_valid(self, field, boundary_value):
        payload = {**VALID_PAYLOAD, field: boundary_value}
        errors = validate_feature_dict(payload)
        field_errors = [e for e in errors if field in e]
        assert field_errors == []

    def test_integer_values_accepted(self):
        payload = {**VALID_PAYLOAD, "temperature": 20, "month": 6}
        assert validate_feature_dict(payload) == []

    def test_partial_payload_flags_missing(self):
        partial = {"temperature": 20.0, "humidity": 60.0}
        errors = validate_feature_dict(partial)
        assert any("missing" in e for e in errors)
