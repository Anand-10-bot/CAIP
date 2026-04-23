from __future__ import annotations

from caip_responses.models.response import Response
from caip_responses.store.conversation_store import ConversationStore


class TestConversationStore:
    def _make_response(
        self, resp_id: str = "resp_1", text: str = "Hello"
    ) -> Response:
        return Response(
            id=resp_id,
            model="test-model",
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "status": "completed",
            }],
        )

    def _make_fc_response(
        self, resp_id: str = "resp_2"
    ) -> Response:
        return Response(
            id=resp_id,
            model="test-model",
            output=[{
                "type": "function_call",
                "id": "item_2",
                "call_id": "fc_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            }],
        )

    def test_save_and_retrieve(self):
        store = ConversationStore()
        response = self._make_response()
        input_items = [{"role": "user", "content": "Hi"}]

        store.save(response, input_items, instructions="Be helpful")

        result = store.get_history("resp_1")
        assert result is not None
        history, instructions = result
        assert instructions == "Be helpful"
        # History = input items + output converted to input
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hi"}
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hello"

    def test_get_history_not_found(self):
        store = ConversationStore()
        assert store.get_history("nonexistent") is None

    def test_has(self):
        store = ConversationStore()
        response = self._make_response()
        store.save(response, [{"role": "user", "content": "Hi"}])

        assert store.has("resp_1") is True
        assert store.has("resp_999") is False

    def test_remove(self):
        store = ConversationStore()
        response = self._make_response()
        store.save(response, [{"role": "user", "content": "Hi"}])

        assert store.remove("resp_1") is True
        assert store.has("resp_1") is False
        assert store.remove("resp_1") is False  # already removed

    def test_clear(self):
        store = ConversationStore()
        for i in range(5):
            resp = self._make_response(f"resp_{i}")
            store.save(resp, [{"role": "user", "content": f"msg {i}"}])

        assert store.size == 5
        store.clear()
        assert store.size == 0

    def test_lru_eviction(self):
        store = ConversationStore(max_size=3)

        for i in range(5):
            resp = self._make_response(f"resp_{i}")
            store.save(resp, [{"role": "user", "content": f"msg {i}"}])

        assert store.size == 3
        # First two should be evicted
        assert store.has("resp_0") is False
        assert store.has("resp_1") is False
        # Last three should remain
        assert store.has("resp_2") is True
        assert store.has("resp_3") is True
        assert store.has("resp_4") is True

    def test_lru_access_refreshes(self):
        store = ConversationStore(max_size=3)

        for i in range(3):
            resp = self._make_response(f"resp_{i}")
            store.save(resp, [{"role": "user", "content": f"msg {i}"}])

        # Access resp_0 to refresh it
        store.get_history("resp_0")

        # Now add a new one — resp_1 should be evicted (oldest untouched)
        resp = self._make_response("resp_3")
        store.save(resp, [{"role": "user", "content": "msg 3"}])

        assert store.has("resp_0") is True  # refreshed
        assert store.has("resp_1") is False  # evicted
        assert store.has("resp_2") is True
        assert store.has("resp_3") is True

    def test_output_to_input_message(self):
        store = ConversationStore()
        response = self._make_response(text="Test output")
        store.save(response, [{"role": "user", "content": "Hi"}])

        history, _ = store.get_history("resp_1")
        # Last item should be the assistant message
        assert history[-1] == {"role": "assistant", "content": "Test output"}

    def test_output_to_input_function_call(self):
        store = ConversationStore()
        response = self._make_fc_response()
        store.save(response, [{"role": "user", "content": "Get weather"}])

        history, _ = store.get_history("resp_2")
        fc_item = history[-1]
        assert fc_item["type"] == "function_call"
        assert fc_item["name"] == "get_weather"
        assert fc_item["call_id"] == "fc_1"

    def test_multi_turn_chain(self):
        """Simulate a multi-turn conversation chain."""
        store = ConversationStore()

        # Turn 1
        resp1 = self._make_response("resp_1", "I can help!")
        store.save(resp1, [{"role": "user", "content": "Hello"}], "Be helpful")

        # Turn 2: use previous_response_id
        history, instructions = store.get_history("resp_1")
        # Client would prepend history to new input
        turn2_input = history + [{"role": "user", "content": "What's 2+2?"}]
        resp2 = self._make_response("resp_2", "4")
        store.save(resp2, turn2_input, instructions)

        # Turn 3: chain from resp_2
        history2, instructions2 = store.get_history("resp_2")
        assert instructions2 == "Be helpful"
        # Should have full chain: user "Hello" + assistant "I can help!" + user "What's 2+2?" + assistant "4"
        assert len(history2) == 4  # turn2 input (3 items) + turn2 output converted to input (1 item)

    def test_size_property(self):
        store = ConversationStore()
        assert store.size == 0

        store.save(self._make_response("r1"), [])
        assert store.size == 1

        store.save(self._make_response("r2"), [])
        assert store.size == 2

    def test_instructions_fallback(self):
        """Instructions from previous turn should be available."""
        store = ConversationStore()
        resp = self._make_response()
        store.save(resp, [{"role": "user", "content": "Hi"}], "Original instructions")

        _, instructions = store.get_history("resp_1")
        assert instructions == "Original instructions"

    def test_no_instructions(self):
        store = ConversationStore()
        resp = self._make_response()
        store.save(resp, [{"role": "user", "content": "Hi"}])

        _, instructions = store.get_history("resp_1")
        assert instructions is None
