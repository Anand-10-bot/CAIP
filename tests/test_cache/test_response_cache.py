from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from caip_responses.cache.response_cache import ResponseCache
from caip_responses.models.response import Response


def _make_response(resp_id: str = "resp_001") -> Response:
    return Response(id=resp_id, model="gpt-4.1", status="completed")


class TestResponseCacheInit:
    def test_defaults(self):
        cache = ResponseCache()
        assert cache.size == 0
        assert cache.enabled is True

    def test_custom_params(self):
        cache = ResponseCache(max_size=100, default_ttl=60, enabled=False)
        assert cache.enabled is False

    def test_enable_disable(self):
        cache = ResponseCache(enabled=False)
        assert cache.enabled is False
        cache.enabled = True
        assert cache.enabled is True


class TestBuildKey:
    def test_deterministic(self):
        cache = ResponseCache()
        k1 = cache.build_key(model="gpt-4.1", input="hello", temperature=0)
        k2 = cache.build_key(model="gpt-4.1", input="hello", temperature=0)
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        cache = ResponseCache()
        k1 = cache.build_key(model="gpt-4.1", input="hello")
        k2 = cache.build_key(model="gpt-4.1", input="world")
        assert k1 != k2

    def test_ignores_stream_and_metadata(self):
        cache = ResponseCache()
        k1 = cache.build_key(model="gpt-4.1", input="hello")
        k2 = cache.build_key(model="gpt-4.1", input="hello", stream=True, metadata={"a": "b"}, user="u1")
        assert k1 == k2

    def test_ignores_none_values(self):
        cache = ResponseCache()
        k1 = cache.build_key(model="gpt-4.1", input="hello")
        k2 = cache.build_key(model="gpt-4.1", input="hello", tools=None)
        assert k1 == k2

    def test_key_is_32_hex_chars(self):
        cache = ResponseCache()
        key = cache.build_key(model="gpt-4.1", input="test")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)


class TestGetSet:
    def test_set_and_get(self):
        cache = ResponseCache()
        resp = _make_response()
        key = cache.build_key(model="gpt-4.1", input="hello")
        cache.set(key, resp)
        assert cache.get(key) is resp
        assert cache.size == 1

    def test_get_miss(self):
        cache = ResponseCache()
        assert cache.get("nonexistent_key") is None

    def test_get_disabled_returns_none(self):
        cache = ResponseCache(enabled=False)
        resp = _make_response()
        key = "some_key"
        cache.set(key, resp)
        assert cache.get(key) is None

    def test_set_disabled_does_not_store(self):
        cache = ResponseCache(enabled=False)
        cache.set("key", _make_response())
        assert cache.size == 0

    def test_custom_ttl(self):
        cache = ResponseCache(default_ttl=3600)
        resp = _make_response()
        cache.set("key", resp, ttl=3600)
        assert cache.get("key") is resp


class TestTTLExpiry:
    def test_expired_entry_returns_none(self):
        cache = ResponseCache(default_ttl=1)
        resp = _make_response()
        cache.set("key", resp)

        # Mock time.monotonic to simulate passage of time
        original_time = time.monotonic()
        with patch("caip_responses.cache.response_cache.time.monotonic", return_value=original_time + 2):
            result = cache.get("key")
        assert result is None

    def test_non_expired_entry_returns_value(self):
        cache = ResponseCache(default_ttl=3600)
        resp = _make_response()
        cache.set("key", resp)
        # Immediate access should work
        assert cache.get("key") is resp

    def test_zero_ttl_never_expires(self):
        cache = ResponseCache(default_ttl=0)
        resp = _make_response()
        cache.set("key", resp)

        original_time = time.monotonic()
        with patch("caip_responses.cache.response_cache.time.monotonic", return_value=original_time + 99999):
            result = cache.get("key")
        assert result is resp


class TestLRUEviction:
    def test_evicts_oldest_when_full(self):
        cache = ResponseCache(max_size=2)
        r1 = _make_response("resp_1")
        r2 = _make_response("resp_2")
        r3 = _make_response("resp_3")

        cache.set("k1", r1)
        cache.set("k2", r2)
        cache.set("k3", r3)

        assert cache.size == 2
        assert cache.get("k1") is None  # evicted (oldest)
        assert cache.get("k2") is r2
        assert cache.get("k3") is r3

    def test_access_promotes_entry(self):
        cache = ResponseCache(max_size=2)
        r1 = _make_response("resp_1")
        r2 = _make_response("resp_2")

        cache.set("k1", r1)
        cache.set("k2", r2)

        # Access k1 to make it most recently used
        cache.get("k1")

        # Add k3 — should evict k2 (now the oldest)
        r3 = _make_response("resp_3")
        cache.set("k3", r3)

        assert cache.get("k1") is r1
        assert cache.get("k2") is None  # evicted
        assert cache.get("k3") is r3

    def test_overwrite_existing_key_no_eviction(self):
        cache = ResponseCache(max_size=2)
        cache.set("k1", _make_response("resp_1"))
        cache.set("k2", _make_response("resp_2"))

        # Overwrite k1 — should not evict anything
        new_resp = _make_response("resp_1_new")
        cache.set("k1", new_resp)
        assert cache.size == 2
        assert cache.get("k1") is new_resp


class TestInvalidateAndClear:
    def test_invalidate_existing(self):
        cache = ResponseCache()
        cache.set("k1", _make_response())
        assert cache.invalidate("k1") is True
        assert cache.get("k1") is None
        assert cache.size == 0

    def test_invalidate_nonexistent(self):
        cache = ResponseCache()
        assert cache.invalidate("nope") is False

    def test_clear(self):
        cache = ResponseCache()
        cache.set("k1", _make_response("r1"))
        cache.set("k2", _make_response("r2"))
        cache.clear()
        assert cache.size == 0


class TestStats:
    def test_initial_stats(self):
        cache = ResponseCache()
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total"] == 0
        assert stats["hit_rate_pct"] == 0
        assert stats["size"] == 0

    def test_stats_after_operations(self):
        cache = ResponseCache()
        cache.set("k1", _make_response())

        cache.get("k1")   # hit
        cache.get("k1")   # hit
        cache.get("k2")   # miss

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["total"] == 3
        assert stats["hit_rate_pct"] == pytest.approx(66.7, abs=0.1)
        assert stats["size"] == 1

    def test_disabled_cache_counts_misses(self):
        cache = ResponseCache(enabled=False)
        cache.get("k1")
        cache.get("k2")
        assert cache.stats["misses"] == 2
        assert cache.stats["hits"] == 0

    def test_reset_stats(self):
        cache = ResponseCache()
        cache.set("k1", _make_response())
        cache.get("k1")
        cache.get("nope")
        cache.reset_stats()
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
