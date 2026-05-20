"""Tests for in-memory TTL cache."""
from __future__ import annotations

import time

import pytest

from app.cache import cache_clear, cache_get, cache_invalidate, cache_set, cache_size


@pytest.fixture(autouse=True)
def clear_cache():
    cache_clear()
    yield
    cache_clear()


class TestCacheSetGet:
    def test_set_and_get(self):
        cache_set("key1", "value1")
        assert cache_get("key1") == "value1"

    def test_missing_key_returns_none(self):
        assert cache_get("nonexistent") is None

    def test_overwrite_key(self):
        cache_set("key", 1)
        cache_set("key", 2)
        assert cache_get("key") == 2

    def test_stores_dict(self):
        cache_set("metrics", {"r2": 0.85})
        assert cache_get("metrics")["r2"] == 0.85

    @pytest.mark.parametrize("value", [0, "", [], False, None])
    def test_stores_falsy_values(self, value):
        cache_set("falsy", value)
        # None means "not set"; other falsy values are stored but
        # we can check size to confirm they were inserted
        if value is not None:
            assert cache_size() == 1


class TestCacheExpiry:
    def test_expired_entry_returns_none(self):
        cache_set("tmp", "v", ttl=0.01)
        time.sleep(0.05)
        assert cache_get("tmp") is None

    def test_non_expired_entry_survives(self):
        cache_set("alive", "yes", ttl=60)
        assert cache_get("alive") == "yes"


class TestCacheInvalidate:
    def test_invalidate_removes_key(self):
        cache_set("k", "v")
        cache_invalidate("k")
        assert cache_get("k") is None

    def test_invalidate_missing_key_noop(self):
        cache_invalidate("ghost")  # should not raise


class TestCacheSize:
    def test_size_increases_on_set(self):
        assert cache_size() == 0
        cache_set("a", 1)
        assert cache_size() == 1
        cache_set("b", 2)
        assert cache_size() == 2
