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
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
