"""Tests for application settings."""
from __future__ import annotations

import os

import pytest

from app.config import Settings, get_settings


class TestSettings:
    def test_default_log_level(self):
        s = Settings()
        assert s.log_level in ("INFO", "DEBUG", "WARNING", "ERROR")

    def test_default_rate_limit_positive(self):
        s = Settings()
        assert s.rate_limit_per_minute > 0

    def test_app_name(self):
        s = Settings()
        assert s.app_name == "Climate-Pulse"

    def test_app_version(self):
        s = Settings()
        assert s.app_version == "1.0.0"

    def test_get_settings_returns_settings(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_debug_default_false(self):
        s = Settings()
        assert s.debug is False

    def test_model_dir_is_string(self):
        s = Settings()
        assert isinstance(s.model_dir, str)
        assert len(s.model_dir) > 0

    def test_database_url_has_default(self):
        s = Settings()
        assert "sqlite" in s.database_url or "postgresql" in s.database_url

    def test_rate_limit_is_int(self):
        s = Settings()
        assert isinstance(s.rate_limit_per_minute, int)

    def test_debug_is_bool(self):
        s = Settings()
        assert isinstance(s.debug, bool)
