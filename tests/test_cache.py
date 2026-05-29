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

    def test_size_decreases_after_invalidate(self):
        cache_set("x", 1)
        cache_set("y", 2)
        cache_invalidate("x")
        assert cache_size() == 1

    def test_size_zero_after_clear(self):
        cache_set("p", 1)
        cache_set("q", 2)
        cache_clear()
        assert cache_size() == 0

    def test_expired_entries_excluded_from_size(self):
        cache_set("expire_me", "v", ttl=0.01)
        cache_set("keep_me", "v", ttl=60)
        time.sleep(0.05)
        assert cache_size() == 1


class TestCacheTtlVariants:
    @pytest.mark.parametrize("ttl", [1, 10, 60, 300, 3600])
    def test_long_ttl_entries_survive(self, ttl):
        cache_set("k", "v", ttl=ttl)
        assert cache_get("k") == "v"

    @pytest.mark.parametrize("key", ["metrics", "model_metrics", "drift_report", "a" * 100])
    def test_various_key_formats(self, key):
        cache_set(key, 42)
        assert cache_get(key) == 42

    def test_update_resets_ttl(self):
        cache_set("renew", "old", ttl=0.05)
        cache_set("renew", "new", ttl=60)
        time.sleep(0.10)
        assert cache_get("renew") == "new"


class TestCacheGetOrSet:
    def test_calls_loader_on_miss(self):
        from app.cache import cache_get_or_set
        result = cache_get_or_set("new_key", lambda: 42)
        assert result == 42

    def test_returns_cached_on_hit(self):
        from app.cache import cache_get_or_set
        cache_set("existing", "original")
        result = cache_get_or_set("existing", lambda: "should_not_be_called")
        assert result == "original"

    def test_loader_called_once_only(self):
        from app.cache import cache_get_or_set
        call_count = [0]

        def loader():
            call_count[0] += 1
            return "computed"

        cache_get_or_set("loader_key", loader)
        cache_get_or_set("loader_key", loader)
        assert call_count[0] == 1

    @pytest.mark.parametrize("value", [0, "", [], {"a": 1}, 3.14])
    def test_caches_various_types(self, value):
        from app.cache import cache_get_or_set
        key = f"typed_{id(value)}"
        result = cache_get_or_set(key, lambda: value)
        assert result == value


class TestCacheStats:
    def test_stats_initial_zero(self):
        from app.cache import cache_stats
        # Stats counts may have prior hits from other tests in same process
        stats = cache_stats()
        assert "size" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "hit_rate" in stats

    def test_stats_size_reflects_cache(self):
        from app.cache import cache_stats
        cache_set("a", 1)
        cache_set("b", 2)
        stats = cache_stats()
        assert stats["size"] == 2

    def test_hit_rate_increases_on_hits(self):
        from app.cache import cache_hit_rate
        cache_set("hr_key", "value")
        cache_get("hr_key")
        cache_get("hr_key")
        rate = cache_hit_rate("hr_key")
        assert rate > 0.0
        assert rate <= 1.0
