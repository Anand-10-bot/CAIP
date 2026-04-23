"""Tests for RedisConversationStore.

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

    def mock_exists(key):
        return 1 if key in store else 0

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
    client.exists = mock_exists
    client.delete = mock_delete
    client.scan = mock_scan
    client.ping = MagicMock(return_value=True)
    client._store = store
    return client


class TestRedisConversationStore:
    @pytest.fixture
    def conv_store(self):
        from caip_responses.store.redis_store import RedisConversationStore
        store = RedisConversationStore.__new__(RedisConversationStore)
        store._client = _make_mock_redis()
        store._prefix = "caip:conv:"
        store._ttl = 86400
        store._max_size = 0
        return store

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

    def test_save_and_get_history(self, conv_store):
        resp = self._make_response()
        input_items = [{"role": "user", "content": "Hi"}]

        conv_store.save(resp, input_items, instructions="Be helpful")
        history = conv_store.get_history("resp_1")

        assert history is not None
        items, instructions = history
        assert instructions == "Be helpful"
        assert len(items) == 2
        assert items[0]["role"] == "user"
        assert items[1]["role"] == "assistant"
        assert items[1]["content"] == "Hello"

    def test_get_history_missing(self, conv_store):
        assert conv_store.get_history("nonexistent") is None

    def test_has(self, conv_store):
        assert conv_store.has("resp_1") is False
        conv_store.save(self._make_response(), [{"role": "user", "content": "Hi"}])
        assert conv_store.has("resp_1") is True

    def test_remove(self, conv_store):
        conv_store.save(self._make_response(), [{"role": "user", "content": "Hi"}])
        assert conv_store.remove("resp_1") is True
        assert conv_store.has("resp_1") is False
        assert conv_store.remove("resp_1") is False

    def test_clear(self, conv_store):
        conv_store.save(
            self._make_response("resp_1"), [{"role": "user", "content": "A"}]
        )
        conv_store.save(
            self._make_response("resp_2"), [{"role": "user", "content": "B"}]
        )
        assert conv_store.size == 2
        conv_store.clear()
        assert conv_store.size == 0

    def test_size(self, conv_store):
        assert conv_store.size == 0
        conv_store.save(
            self._make_response("resp_1"), [{"role": "user", "content": "A"}]
        )
        assert conv_store.size == 1

    def test_function_call_in_output(self, conv_store):
        resp = Response(
            id="resp_fc",
            model="test-model",
            output=[{
                "type": "function_call",
                "id": "item_fc",
                "call_id": "fc_1",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            }],
        )
        conv_store.save(resp, [{"role": "user", "content": "Weather?"}])
        history = conv_store.get_history("resp_fc")

        assert history is not None
        items, _ = history
        assert len(items) == 2
        assert items[1]["type"] == "function_call"
        assert items[1]["name"] == "get_weather"

    def test_get_chain_delegates_to_get_history(self, conv_store):
        conv_store.save(
            self._make_response(), [{"role": "user", "content": "Hi"}]
        )
        chain_result = conv_store.get_chain("resp_1")
        history_result = conv_store.get_history("resp_1")
        assert chain_result == history_result

    def test_ping(self, conv_store):
        assert conv_store.ping() is True

    def test_serialization_roundtrip(self, conv_store):
        """Ensure data survives JSON serialization."""
        input_items = [
            {"role": "user", "content": "complex\nmultiline\ninput"},
            {"role": "assistant", "content": "previous turn"},
            {"role": "user", "content": 'with "quotes" and special chars'},
        ]
        resp = self._make_response(text="response with unicode: \u2603")
        conv_store.save(resp, input_items, instructions="instruct: be\n\nhelpful")

        history = conv_store.get_history("resp_1")
        assert history is not None
        items, instructions = history
        assert instructions == "instruct: be\n\nhelpful"
        assert items[0]["content"] == "complex\nmultiline\ninput"

    def test_save_overwrites_existing(self, conv_store):
        conv_store.save(
            self._make_response("resp_1", "First"),
            [{"role": "user", "content": "Q1"}],
        )
        conv_store.save(
            self._make_response("resp_1", "Second"),
            [{"role": "user", "content": "Q2"}],
        )
        history = conv_store.get_history("resp_1")
        assert history is not None
        items, _ = history
        assert items[1]["content"] == "Second"
