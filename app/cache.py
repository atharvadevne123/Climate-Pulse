"""Simple in-memory TTL cache for model metrics and frequent lookups.

Thread-unsafe by design — this service is single-process (one uvicorn worker).
For multi-process deployments, replace with Redis or Memcached.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[Any, float]] = {}
DEFAULT_TTL: float = 300.0  # 5 minutes


def cache_set(key: str, value: Any, ttl: float = DEFAULT_TTL) -> None:
    """Store a value in the cache with an expiry timestamp.

    Args:
        key: Cache key string.
        value: Arbitrary Python value to cache.
        ttl: Time-to-live in seconds (default 300).
    """
    _cache[key] = (value, time.monotonic() + ttl)
    logger.debug("cache.set: key=%s ttl=%.0fs", key, ttl)


def cache_get(key: str) -> Any | None:
    """Return the cached value if it exists and has not expired.

    Args:
        key: Cache key to look up.

    Returns:
        Cached value, or None if missing or expired.
    """
    entry = _cache.get(key)
    if entry is None:
        return None
    value, expiry = entry
    if time.monotonic() > expiry:
        del _cache[key]
        logger.debug("cache.get: key=%s EXPIRED", key)
        return None
    return value


def cache_invalidate(key: str) -> None:
    """Remove a specific key from the cache (no-op if absent).

    Args:
        key: Cache key to remove.
    """
    _cache.pop(key, None)
    logger.debug("cache.invalidate: key=%s", key)


def cache_clear() -> None:
    """Remove all entries from the cache."""
    _cache.clear()
    logger.info("cache.clear: all entries removed")


def cache_size() -> int:
    """Return the count of live (non-expired) cache entries.

    Returns:
        Integer count of non-expired cache entries.
    """
    now = time.monotonic()
    return sum(1 for _, (_, exp) in _cache.items() if now <= exp)
