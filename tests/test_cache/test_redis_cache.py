"""Tests for RedisResponseCache.

Uses a mock Redis client so no actual Redis server is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from caip_responses.models.response import Response


def _make_mock_redis():
    """Create a mock Redis client backed by an in-memory dict."""
    store = {}
    client = MagicMock()

    def mock_set(key, value):
        store[key] = value

    def mock_setex(key, ttl, value):
        store[key] = value

    def mock_get(key):
        return store.get(key)

    def mock_delete(*keys):
        count = 0
        for k in keys:
            if k in store:
                del store[k]
                count += 1
        return count

    def mock_scan(cursor=0, match="*", count=100):
        import fnmatch
        matching = [k for k in store if fnmatch.fnmatch(k, match)]
        return (0, matching)

    client.set = mock_set
    client.setex = mock_setex
    client.get = mock_get
    client.delete = mock_delete
    client.scan = mock_scan
    client.ping = MagicMock(return_value=True)
    client._store = store
    return client


class TestRedisResponseCache:
    @pytest.fixture
    def cache(self):
        from caip_responses.cache.redis_cache import RedisResponseCache
        c = RedisResponseCache.__new__(RedisResponseCache)
        c._client = _make_mock_redis()
        c._prefix = "caip:cache:"
        c._max_size = 0
        c._default_ttl = 3600
        c._enabled = True
        c._hits = 0
        c._misses = 0
        return c

    def _make_response(self, resp_id="resp_1", text="Hello"):
        return Response(
            id=resp_id,
            model="test-model",
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": text, "annotations": []}
                ],
                "status": "completed",
            }],
        )

    def test_build_key_deterministic(self, cache):
        k1 = cache.build_key(model="gpt-4", input="hello", temperature=0)
        k2 = cache.build_key(model="gpt-4", input="hello", temperature=0)
        assert k1 == k2

    def test_build_key_different_inputs(self, cache):
        k1 = cache.build_key(model="gpt-4", input="hello")
        k2 = cache.build_key(model="gpt-4", input="world")
        assert k1 != k2

    def test_build_key_ignores_stream(self, cache):
        k1 = cache.build_key(model="gpt-4", input="hello", stream=True)
        k2 = cache.build_key(model="gpt-4", input="hello", stream=False)
        assert k1 == k2

    def test_set_and_get(self, cache):
        resp = self._make_response()
        key = cache.build_key(model="test", input="hi")

        cache.set(key, resp)
        cached = cache.get(key)

        assert cached is not None
        assert cached.id == "resp_1"
        assert cached.model == "test-model"

    def test_get_miss(self, cache):
        assert cache.get("nonexistent") is None

    def test_get_disabled(self, cache):
        resp = self._make_response()
        key = cache.build_key(model="test", input="hi")
        cache.set(key, resp)

        cache.enabled = False
        assert cache.get(key) is None

    def test_set_disabled(self, cache):
        cache.enabled = False
        resp = self._make_response()
        key = cache.build_key(model="test", input="hi")
        cache.set(key, resp)

        cache.enabled = True
        assert cache.get(key) is None  # was never actually stored

    def test_invalidate(self, cache):
        resp = self._make_response()
        key = cache.build_key(model="test", input="hi")
        cache.set(key, resp)

        assert cache.invalidate(key) is True
        assert cache.get(key) is None
        assert cache.invalidate(key) is False

    def test_clear(self, cache):
        for i in range(3):
            key = cache.build_key(model="test", input=f"msg_{i}")
            cache.set(key, self._make_response(f"resp_{i}"))

        assert cache.size == 3
        cache.clear()
        assert cache.size == 0

    def test_size(self, cache):
        assert cache.size == 0
        key = cache.build_key(model="test", input="hi")
        cache.set(key, self._make_response())
        assert cache.size == 1

    def test_stats(self, cache):
        key = cache.build_key(model="test", input="hi")
        cache.set(key, self._make_response())

        cache.get(key)       # hit
        cache.get("miss_1")  # miss
        cache.get("miss_2")  # miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["total"] == 3
        assert stats["hit_rate_pct"] == pytest.approx(33.3, abs=0.1)

    def test_reset_stats(self, cache):
        cache.get("anything")  # miss
        assert cache.stats["total"] == 1
        cache.reset_stats()
        assert cache.stats["total"] == 0

    def test_response_serialization_roundtrip(self, cache):
        """Response survives JSON serialization to/from Redis."""
        resp = Response(
            id="resp_complex",
            model="test-model",
            status="completed",
            output=[
                {
                    "type": "message",
                    "id": "item_1",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello world!", "annotations": []}
                    ],
                    "status": "completed",
                },
                {
                    "type": "function_call",
                    "id": "item_2",
                    "call_id": "fc_1",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
            ],
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )

        key = cache.build_key(model="test", input="complex")
        cache.set(key, resp)
        cached = cache.get(key)

        assert cached is not None
        assert cached.id == "resp_complex"
        assert len(cached.output) == 2

    def test_ping(self, cache):
        assert cache.ping() is True

    def test_enabled_property(self, cache):
        assert cache.enabled is True
        cache.enabled = False
        assert cache.enabled is False
