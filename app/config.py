"""Application settings via environment variables."""

from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    """Central settings object populated from environment variables."""

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./climate_pulse.db")
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "200"))
    app_name: str = "Climate-Pulse"
    app_version: str = "1.2.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    db_max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    db_pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
