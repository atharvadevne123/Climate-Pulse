"""Tests for application settings."""
from __future__ import annotations

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

    def test_get_settings_returns_settings(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
