from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from caip_responses.client.async_client import AsyncClient
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider


class _TrackingProvider(BaseProvider):
    """Provider that tracks requests and returns configurable responses."""

    def __init__(self, name: str = "anthropic") -> None:
        self._name = name
        self.requests: list[CreateResponseRequest] = []
        self._response_queue: list[Response] = []

    @property
    def provider_name(self) -> str:
        return self._name

    def supports_tool(self, tool_type: str) -> bool:
        return True

    def supports_reasoning(self) -> bool:
        return True

    def queue_response(self, response: Response) -> None:
        self._response_queue.append(response)

    async def create_response(self, request: CreateResponseRequest) -> Response:
        self.requests.append(request)
        if self._response_queue:
            return self._response_queue.pop(0)
        return Response(
            id=f"resp_{len(self.requests)}",
            model=request.model,
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Default response", "annotations": []}],
                "status": "completed",
            }],
        )

    async def create_response_stream(self, request: CreateResponseRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        resp = await self.create_response(
            request.model_copy(update={})  # avoid double-recording
        )
        # Remove the duplicate from requests
        self.requests.pop()

        yield StreamEvent(
            type="response.created",
            sequence_number=0,
            response={"id": resp.id, "model": resp.model, "status": "in_progress"},
        )
        yield StreamEvent(
            type="response.completed",
            sequence_number=1,
            response={"id": resp.id, "model": resp.model, "status": "completed"},
        )


class TestPreviousResponseId:
    @pytest.mark.asyncio
    async def test_stores_response_for_later_retrieval(self):
        """Non-streaming calls should store responses automatically."""
        provider = _TrackingProvider()
        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        resp = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Hello",
            instructions="Be helpful",
        )

        assert client.conversation_store.has(resp.id)

    @pytest.mark.asyncio
    async def test_previous_response_id_prepends_history(self):
        """Using previous_response_id should prepend conversation history."""
        provider = _TrackingProvider()
        provider.queue_response(Response(
            id="resp_first",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I'm here to help!", "annotations": []}],
                "status": "completed",
            }],
        ))
        provider.queue_response(Response(
            id="resp_second",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message",
                "id": "item_2",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "4", "annotations": []}],
                "status": "completed",
            }],
        ))

        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        # First turn
        resp1 = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Hello",
            instructions="Be helpful",
        )
        assert resp1.id == "resp_first"

        # Second turn with previous_response_id
        await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="What's 2+2?",
            previous_response_id="resp_first",
        )

        # The second request should have the full history prepended
        second_request = provider.requests[1]
        input_items = second_request.input
        assert isinstance(input_items, list)
        # Should contain: original "Hello" + assistant "I'm here to help!" + new "What's 2+2?"
        assert len(input_items) == 3
        assert input_items[0] == {"role": "user", "content": "Hello"}
        assert input_items[1]["role"] == "assistant"
        assert "help" in input_items[1]["content"]
        assert input_items[2] == {"role": "user", "content": "What's 2+2?"}

    @pytest.mark.asyncio
    async def test_previous_response_id_preserves_instructions(self):
        """Instructions from the first turn should carry forward."""
        provider = _TrackingProvider()
        provider.queue_response(Response(
            id="resp_A",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                "status": "completed",
            }],
        ))
        provider.queue_response(Response(
            id="resp_B",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message",
                "id": "item_2",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Sure", "annotations": []}],
                "status": "completed",
            }],
        ))

        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Hi",
            instructions="Always respond in French",
        )

        # Second turn without explicit instructions
        await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="What's up?",
            previous_response_id="resp_A",
        )

        second_request = provider.requests[1]
        assert second_request.instructions == "Always respond in French"

    @pytest.mark.asyncio
    async def test_previous_response_id_missing_returns_without_history(self):
        """If previous_response_id not in store, just proceed without history."""
        provider = _TrackingProvider()
        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        resp = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Hello",
            previous_response_id="nonexistent_id",
        )

        # Should still work — just without prepended history
        assert resp.output_text == "Default response"
        first_request = provider.requests[0]
        assert first_request.input == "Hello"

    @pytest.mark.asyncio
    async def test_store_disabled_does_not_save(self):
        """store=False should skip saving."""
        provider = _TrackingProvider()
        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        resp = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Hello",
            store=False,
        )

        assert not client.conversation_store.has(resp.id)

    @pytest.mark.asyncio
    async def test_streaming_stores_response(self):
        """Streaming should also store the response on completion."""
        provider = _TrackingProvider()
        provider.queue_response(Response(
            id="resp_stream_1",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Streamed!", "annotations": []}],
                "status": "completed",
            }],
        ))

        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        stream = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Hello",
            stream=True,
        )
        # Must consume the stream for store to capture
        async for _ in stream:
            pass

        assert client.conversation_store.has("resp_stream_1")


class TestAgenticLoopIntegration:
    @pytest.mark.asyncio
    async def test_auto_tool_execution(self):
        """Client should auto-run the agentic loop when handlers are registered."""
        provider = _TrackingProvider()
        # First call returns a function call
        provider.queue_response(Response(
            id="resp_fc",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "function_call",
                "id": "item_1",
                "call_id": "fc_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            }],
        ))
        # Second call returns text (after tool output)
        provider.queue_response(Response(
            id="resp_final",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message",
                "id": "item_2",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "It's 72F in SF", "annotations": []}],
                "status": "completed",
            }],
        ))

        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        # Register a tool handler
        async def weather_handler(name: str, args: dict) -> str:
            return json.dumps({"temp": 72, "city": args.get("city", "")})

        client.tools.register("get_weather", weather_handler)

        response = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="What's the weather in SF?",
            tools=[{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        )

        # Should get the final text response (loop ran automatically)
        assert response.output_text == "It's 72F in SF"
        # Provider should have been called twice
        assert len(provider.requests) == 2

    @pytest.mark.asyncio
    async def test_no_handlers_skips_loop(self):
        """Without registered handlers, function calls pass through without looping."""
        provider = _TrackingProvider()
        provider.queue_response(Response(
            id="resp_fc",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "function_call",
                "id": "item_1",
                "call_id": "fc_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            }],
        ))

        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        # No tool handlers registered
        response = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="What's the weather?",
            tools=[{
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            }],
        )

        # Should return the function call directly (no loop)
        assert response.has_function_calls is True
        assert len(provider.requests) == 1


class TestStructuredOutputValidation:
    @pytest.mark.asyncio
    async def test_valid_json_output_no_error(self):
        """Valid JSON output matching schema should not produce an error."""
        provider = _TrackingProvider()
        provider.queue_response(Response(
            id="resp_json",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": '{"name": "Alice", "age": 30}', "annotations": []}],
                "status": "completed",
            }],
        ))

        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        response = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Give me user data",
            text={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                    },
                },
            },
        )

        assert response.output_text == '{"name": "Alice", "age": 30}'
        assert response.error is None

    @pytest.mark.asyncio
    async def test_invalid_json_output_sets_error(self):
        """Non-JSON output with a json_schema request should set an error."""
        provider = _TrackingProvider()
        provider.queue_response(Response(
            id="resp_bad",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "This is not JSON!", "annotations": []}],
                "status": "completed",
            }],
        ))

        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        response = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Give me data",
            text={
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object"},
                },
            },
        )

        assert response.output_text == "This is not JSON!"
        assert response.error is not None
        assert response.error["type"] == "json_schema_validation_error"

    @pytest.mark.asyncio
    async def test_no_text_config_skips_validation(self):
        """Without text config, no validation is performed."""
        provider = _TrackingProvider()
        client = AsyncClient(providers={"anthropic": provider})
        client._registry.add_prefix_mapping("claude-", "anthropic")

        response = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Hello",
        )

        assert response.error is None
