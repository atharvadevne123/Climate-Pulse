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


class TestSettingsEdgeCases:
    def test_settings_database_url_string(self):
        from app.config import Settings
        s = Settings()
        assert isinstance(s.database_url, str)

    def test_settings_model_dir_default(self):
        import os

        from app.config import Settings
        os.environ.setdefault("MODEL_DIR", "./models")
        s = Settings()
        assert s.model_dir is not None

    def test_get_settings_returns_same_instance(self):
        from app.config import get_settings
        # Cache means same instance returned
        assert get_settings() is get_settings()

    def test_settings_app_name_non_empty(self):
        from app.config import Settings
        s = Settings()
        assert len(s.app_name) > 0

    def test_settings_rate_limit_at_least_1(self):
        from app.config import Settings
        s = Settings()
        assert s.rate_limit_per_minute >= 1


class TestSettingsPoolConfig:
    def test_db_pool_size_positive(self):
        from app.config import Settings
        s = Settings()
        assert s.db_pool_size > 0

    def test_db_max_overflow_non_negative(self):
        from app.config import Settings
        s = Settings()
        assert s.db_max_overflow >= 0

    def test_db_pool_timeout_positive(self):
        from app.config import Settings
        s = Settings()
        assert s.db_pool_timeout > 0

    def test_db_pool_size_is_int(self):
        from app.config import Settings
        s = Settings()
        assert isinstance(s.db_pool_size, int)

    def test_db_max_overflow_is_int(self):
        from app.config import Settings
        s = Settings()
        assert isinstance(s.db_max_overflow, int)

    def test_app_version_is_1_2_0(self):
        from app.config import Settings
        s = Settings()
        assert s.app_version == "1.2.0"

    @pytest.mark.parametrize("field", ["db_pool_size", "db_max_overflow", "db_pool_timeout"])
    def test_pool_fields_exist(self, field):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, field)
