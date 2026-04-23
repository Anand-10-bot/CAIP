"""Cross-provider integration tests.

These tests verify that the library works correctly when multiple providers
are configured and used together — provider switching, streaming, conversation
store, parallel tool execution, and structured output enforcement.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from caip_responses.client.async_client import AsyncClient
from caip_responses.models.common import Usage
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider

# ---------------------------------------------------------------------------
# Configurable mock provider
# ---------------------------------------------------------------------------


class _FakeProvider(BaseProvider):
    """Mock provider with configurable responses, streaming, and tool support."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.requests: list[CreateResponseRequest] = []
        self._responses: list[Response] = []
        self._stream_events: list[list[StreamEvent]] | None = None
        self._supported_tools: set[str] = {"function"}

    @property
    def provider_name(self) -> str:
        return self._name

    def supports_tool(self, tool_type: str) -> bool:
        return tool_type in self._supported_tools

    def supports_reasoning(self) -> bool:
        return True

    def queue_response(self, resp: Response) -> None:
        self._responses.append(resp)

    def queue_stream(self, events: list[StreamEvent]) -> None:
        if self._stream_events is None:
            self._stream_events = []
        self._stream_events.append(events)

    async def create_response(self, request: CreateResponseRequest) -> Response:
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        return Response(
            id=f"resp_{self._name}_{len(self.requests)}",
            model=request.model,
            output=[{
                "type": "message",
                "id": "item_1",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": f"Response from {self._name}",
                    "annotations": [],
                }],
                "status": "completed",
            }],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if self._stream_events:
            events = self._stream_events.pop(0)
            for event in events:
                yield event
            return

        resp_id = f"resp_{self._name}_{len(self.requests)}"
        yield StreamEvent(
            type="response.created",
            sequence_number=0,
            response={"id": resp_id, "model": request.model, "status": "in_progress"},
        )
        yield StreamEvent(
            type="response.output_item.added",
            sequence_number=1,
            output_index=0,
            item={"type": "message", "id": "item_1", "role": "assistant"},
        )
        yield StreamEvent(
            type="response.content_part.added",
            sequence_number=2,
            output_index=0,
            content_index=0,
            part={"type": "output_text", "text": ""},
        )
        yield StreamEvent(
            type="response.output_text.delta",
            sequence_number=3,
            output_index=0,
            content_index=0,
            delta=f"Streamed from {self._name}",
        )
        yield StreamEvent(
            type="response.output_text.done",
            sequence_number=4,
            output_index=0,
            content_index=0,
            text=f"Streamed from {self._name}",
        )
        yield StreamEvent(
            type="response.content_part.done",
            sequence_number=5,
            output_index=0,
            content_index=0,
        )
        yield StreamEvent(
            type="response.output_item.done",
            sequence_number=6,
            output_index=0,
        )
        yield StreamEvent(
            type="response.completed",
            sequence_number=7,
            response={
                "id": resp_id,
                "model": request.model,
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )


def _make_client(**providers: _FakeProvider) -> AsyncClient:
    """Create an AsyncClient with fake providers registered."""
    client = AsyncClient(
        providers=providers,
        discover_plugins=False,
    )
    for name in providers:
        if name == "anthropic":
            client._registry.add_prefix_mapping("claude-", "anthropic")
        elif name == "gemini":
            client._registry.add_prefix_mapping("gemini-", "gemini")
        elif name == "sarvam":
            client._registry.add_prefix_mapping("sarvam-", "sarvam")
    return client


# ---------------------------------------------------------------------------
# Provider switching
# ---------------------------------------------------------------------------


class TestProviderSwitching:
    @pytest.mark.asyncio
    async def test_switch_provider_by_model_prefix(self):
        """Changing the model name routes to the correct provider."""
        anthropic = _FakeProvider("anthropic")
        gemini = _FakeProvider("gemini")
        client = _make_client(anthropic=anthropic, gemini=gemini)

        resp1 = await client.responses.create(
            model="claude-sonnet-4-20250514", input="Hello"
        )
        assert resp1.output_text == "Response from anthropic"
        assert len(anthropic.requests) == 1
        assert len(gemini.requests) == 0

        resp2 = await client.responses.create(
            model="gemini-2.0-flash", input="Hello"
        )
        assert resp2.output_text == "Response from gemini"
        assert len(gemini.requests) == 1

    @pytest.mark.asyncio
    async def test_explicit_provider_override(self):
        """provider= param overrides model-prefix routing."""
        anthropic = _FakeProvider("anthropic")
        gemini = _FakeProvider("gemini")
        client = _make_client(anthropic=anthropic, gemini=gemini)

        resp = await client.responses.create(
            model="custom-model", input="Hello", provider="gemini"
        )
        assert resp.output_text == "Response from gemini"
        assert len(gemini.requests) == 1
        assert len(anthropic.requests) == 0

    @pytest.mark.asyncio
    async def test_same_api_different_providers(self):
        """Identical API call works across providers — core principle."""
        anthropic = _FakeProvider("anthropic")
        gemini = _FakeProvider("gemini")
        sarvam = _FakeProvider("sarvam")
        client = _make_client(anthropic=anthropic, gemini=gemini, sarvam=sarvam)

        for model, expected_provider in [
            ("claude-sonnet-4-20250514", "anthropic"),
            ("gemini-2.0-flash", "gemini"),
            ("sarvam-2b", "sarvam"),
        ]:
            resp = await client.responses.create(
                model=model,
                input="What is 2+2?",
                instructions="Be concise",
                temperature=0.5,
            )
            assert resp.output_text == f"Response from {expected_provider}"


# ---------------------------------------------------------------------------
# Streaming across providers
# ---------------------------------------------------------------------------


class TestCrossProviderStreaming:
    @pytest.mark.asyncio
    async def test_streaming_canonical_events(self):
        """All providers emit the same canonical event sequence."""
        for name, prefix in [("anthropic", "claude-"), ("gemini", "gemini-")]:
            provider = _FakeProvider(name)
            client = _make_client(**{name: provider})

            stream = await client.responses.create(
                model=f"{prefix}test-model", input="Hi", stream=True
            )
            events = [e async for e in stream]
            types = [e.type for e in events]

            assert types[0] == "response.created"
            assert "response.output_text.delta" in types
            assert types[-1] == "response.completed"

    @pytest.mark.asyncio
    async def test_streaming_stores_response(self):
        """Streaming responses are stored for previous_response_id use."""
        anthropic = _FakeProvider("anthropic")
        client = _make_client(anthropic=anthropic)

        stream = await client.responses.create(
            model="claude-test", input="Hello", stream=True
        )
        async for _ in stream:
            pass

        assert client.conversation_store.has("resp_anthropic_1")


# ---------------------------------------------------------------------------
# Conversation store across providers
# ---------------------------------------------------------------------------


class TestCrossProviderConversation:
    @pytest.mark.asyncio
    async def test_conversation_chain_single_provider(self):
        """Multi-turn conversation works with previous_response_id."""
        anthropic = _FakeProvider("anthropic")
        anthropic.queue_response(Response(
            id="resp_turn1",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message", "id": "item_1", "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there!", "annotations": []}],
                "status": "completed",
            }],
            usage=Usage(input_tokens=5, output_tokens=3, total_tokens=8),
        ))
        anthropic.queue_response(Response(
            id="resp_turn2",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message", "id": "item_2", "role": "assistant",
                "content": [{"type": "output_text", "text": "4", "annotations": []}],
                "status": "completed",
            }],
            usage=Usage(input_tokens=15, output_tokens=1, total_tokens=16),
        ))

        client = _make_client(anthropic=anthropic)

        resp1 = await client.responses.create(
            model="claude-sonnet-4-20250514", input="Hello"
        )
        resp2 = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="What's 2+2?",
            previous_response_id=resp1.id,
        )

        assert resp2.output_text == "4"
        # Second request should have full history
        second_input = anthropic.requests[1].input
        assert isinstance(second_input, list)
        assert len(second_input) >= 3  # user + assistant + user


# ---------------------------------------------------------------------------
# Parallel tool execution
# ---------------------------------------------------------------------------


class TestParallelToolExecution:
    @pytest.mark.asyncio
    async def test_parallel_tool_calls(self):
        """Multiple tool calls in one response execute in parallel."""
        anthropic = _FakeProvider("anthropic")
        # First call returns two function calls
        anthropic.queue_response(Response(
            id="resp_fc",
            model="claude-sonnet-4-20250514",
            output=[
                {
                    "type": "function_call", "id": "item_1",
                    "call_id": "fc_1", "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                },
                {
                    "type": "function_call", "id": "item_2",
                    "call_id": "fc_2", "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
            ],
        ))
        # Second call returns final text
        anthropic.queue_response(Response(
            id="resp_final",
            model="claude-sonnet-4-20250514",
            output=[{
                "type": "message", "id": "item_3", "role": "assistant",
                "content": [{"type": "output_text", "text": "SF: 72F, NYC: 65F", "annotations": []}],
                "status": "completed",
            }],
            usage=Usage(input_tokens=30, output_tokens=10, total_tokens=40),
        ))

        client = _make_client(anthropic=anthropic)
        call_order: list[str] = []

        async def weather_handler(name: str, args: dict) -> str:
            city = args.get("city", "")
            call_order.append(city)
            return json.dumps({"temp": 72 if city == "SF" else 65, "city": city})

        client.tools.register("get_weather", weather_handler)

        response = await client.responses.create(
            model="claude-sonnet-4-20250514",
            input="Weather in SF and NYC?",
            tools=[{
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        )

        assert response.output_text == "SF: 72F, NYC: 65F"
        # Both tools were called
        assert len(call_order) == 2
        assert set(call_order) == {"SF", "NYC"}


# ---------------------------------------------------------------------------
# Structured output across providers
# ---------------------------------------------------------------------------


class TestCrossProviderStructuredOutput:
    @pytest.mark.asyncio
    async def test_valid_json_passes_validation(self):
        """Valid JSON matching schema passes for any provider."""
        for name, prefix in [("anthropic", "claude-"), ("gemini", "gemini-")]:
            provider = _FakeProvider(name)
            provider.queue_response(Response(
                id=f"resp_json_{name}",
                model=f"{prefix}model",
                output=[{
                    "type": "message", "id": "item_1", "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": '{"name": "Alice", "age": 30}',
                        "annotations": [],
                    }],
                    "status": "completed",
                }],
                usage=Usage(input_tokens=10, output_tokens=8, total_tokens=18),
            ))

            client = _make_client(**{name: provider})
            response = await client.responses.create(
                model=f"{prefix}model",
                input="Give user data",
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
            assert response.error is None

    @pytest.mark.asyncio
    async def test_invalid_json_fails_validation(self):
        """Invalid JSON output sets error regardless of provider."""
        provider = _FakeProvider("anthropic")
        provider.queue_response(Response(
            id="resp_bad",
            model="claude-model",
            output=[{
                "type": "message", "id": "item_1", "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Not valid JSON",
                    "annotations": [],
                }],
                "status": "completed",
            }],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        ))

        client = _make_client(anthropic=provider)
        response = await client.responses.create(
            model="claude-model",
            input="Give data",
            text={"format": {"type": "json_schema", "schema": {"type": "object"}}},
        )
        assert response.error is not None
        assert response.error["type"] == "json_schema_validation_error"


# ---------------------------------------------------------------------------
# Cost tracking across providers
# ---------------------------------------------------------------------------


class TestCrossProviderCostTracking:
    @pytest.mark.asyncio
    async def test_cost_tracked_per_provider(self):
        """Costs are tracked separately per provider."""
        from caip_responses.cost.tracker import ModelPricing

        anthropic = _FakeProvider("anthropic")
        gemini = _FakeProvider("gemini")
        client = _make_client(anthropic=anthropic, gemini=gemini)

        client.cost_tracker.set_pricing(
            "claude-model",
            ModelPricing(input_cost_per_million=3.0, output_cost_per_million=15.0),
        )
        client.cost_tracker.set_pricing(
            "gemini-model",
            ModelPricing(input_cost_per_million=0.5, output_cost_per_million=1.5),
        )

        await client.responses.create(model="claude-model", input="Hello")
        await client.responses.create(model="gemini-model", input="Hello")

        assert client.cost_tracker.total_requests == 2
        assert client.cost_tracker.total_cost > 0

        # Provider-level breakdown
        breakdown = client.cost_tracker.by_model()
        assert "claude-model" in breakdown
        assert "gemini-model" in breakdown
        assert breakdown["claude-model"].total_cost_usd > breakdown["gemini-model"].total_cost_usd
