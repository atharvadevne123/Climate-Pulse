"""Simple in-memory TTL cache for model metrics and frequent lookups.

Thread-unsafe by design — this service is single-process (one uvicorn worker).
For multi-process deployments, replace with Redis or Memcached.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[Any, float]] = {}
DEFAULT_TTL: float = 300.0  # 5 minutes

_hits: defaultdict[str, int] = defaultdict(int)
_misses: defaultdict[str, int] = defaultdict(int)


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
        _misses[key] += 1
        return None
    value, expiry = entry
    if time.monotonic() > expiry:
        del _cache[key]
        _misses[key] += 1
        logger.debug("cache.get: key=%s EXPIRED", key)
        return None
    _hits[key] += 1
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


def cache_get_or_set(key: str, loader: Callable[[], Any], ttl: float = DEFAULT_TTL) -> Any:
    """Return cached value if present; otherwise call *loader*, cache, and return the result.

    Args:
        key: Cache key string.
        loader: Zero-argument callable that computes the value on a cache miss.
        ttl: Time-to-live in seconds for newly cached entries (default 300).

    Returns:
        Cached or freshly computed value.
    """
    value = cache_get(key)
    if value is None:
        value = loader()
        cache_set(key, value, ttl=ttl)
    return value


def cache_hit_rate(key: str) -> float:
    """Return the hit rate (0.0–1.0) for a given cache key since process start.

    Args:
        key: Cache key to query.

    Returns:
        Float in [0.0, 1.0]; returns 0.0 if the key has never been accessed.
    """
    total = _hits[key] + _misses[key]
    if total == 0:
        return 0.0
    return _hits[key] / total


def cache_stats() -> dict[str, Any]:
    """Return aggregate hit/miss statistics and current cache size.

    Returns:
        Dict with keys ``size``, ``total_hits``, ``total_misses``, ``hit_rate``.
    """
    total_hits = sum(_hits.values())
    total_misses = sum(_misses.values())
    total = total_hits + total_misses
    return {
        "size": cache_size(),
        "total_hits": total_hits,
        "total_misses": total_misses,
        "hit_rate": round(total_hits / total, 4) if total > 0 else 0.0,
    }
