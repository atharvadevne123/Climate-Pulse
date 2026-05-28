"""Tests for app/utils.py utility functions."""
from __future__ import annotations

import pytest

from app.utils import clamp, flatten_dict, round_to_sig_figs, safe_float


class TestClamp:
    def test_value_within_range_unchanged(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_value_below_lo_clamped_to_lo(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_value_above_hi_clamped_to_hi(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_value_at_lo_boundary(self):
        assert clamp(0.0, 0.0, 10.0) == 0.0

    def test_value_at_hi_boundary(self):
        assert clamp(10.0, 0.0, 10.0) == 10.0

    @pytest.mark.parametrize("value,lo,hi,expected", [
        (-100.0, -90.0, 60.0, -90.0),
        (70.0, -90.0, 60.0, 60.0),
        (20.0, -90.0, 60.0, 20.0),
    ])
    def test_temperature_range_clamping(self, value, lo, hi, expected):
        assert clamp(value, lo, hi) == expected


class TestRoundToSigFigs:
    def test_zero_returns_zero(self):
        assert round_to_sig_figs(0.0) == 0.0

    def test_positive_value(self):
        result = round_to_sig_figs(3.14159, 3)
        assert result == pytest.approx(3.14, rel=1e-3)

    def test_large_value(self):
        result = round_to_sig_figs(123456.789, 4)
        assert result == pytest.approx(123500.0, rel=1e-3)

    def test_small_value(self):
        result = round_to_sig_figs(0.001234, 2)
        assert result == pytest.approx(0.0012, rel=1e-3)

    def test_negative_value(self):
        result = round_to_sig_figs(-9.876, 3)
        assert result == pytest.approx(-9.88, rel=1e-3)

    def test_default_4_sig_figs(self):
        result = round_to_sig_figs(1.23456789)
        assert result == pytest.approx(1.235, rel=1e-3)


class TestFlattenDict:
    def test_flat_dict_unchanged(self):
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_nested_dict_flattened(self):
        d = {"a": {"b": {"c": 3}}}
        assert flatten_dict(d) == {"a.b.c": 3}

    def test_mixed_depth(self):
        d = {"x": 1, "y": {"z": 2}}
        result = flatten_dict(d)
        assert result["x"] == 1
        assert result["y.z"] == 2

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        result = flatten_dict(d, sep="/")
        assert "a/b" in result

    def test_empty_dict(self):
        assert flatten_dict({}) == {}

    def test_list_values_not_expanded(self):
        d = {"a": [1, 2, 3]}
        result = flatten_dict(d)
        assert result["a"] == [1, 2, 3]


class TestSafeFloat:
    def test_float_passthrough(self):
        assert safe_float(3.14) == pytest.approx(3.14)

    def test_int_converted(self):
        assert safe_float(5) == pytest.approx(5.0)

    def test_string_number_converted(self):
        assert safe_float("2.5") == pytest.approx(2.5)

    def test_invalid_string_returns_default(self):
        assert safe_float("abc") == 0.0

    def test_none_returns_default(self):
        assert safe_float(None) == 0.0

    def test_custom_default_returned(self):
        assert safe_float("bad", default=-1.0) == pytest.approx(-1.0)

    def test_zero_string(self):
        assert safe_float("0") == pytest.approx(0.0)
