"""Integration tests for the two-tier dispatch in AgentLoop.

Tests that builtin tool handlers are called for synthetic function names,
while user-registered handlers are called for regular function names.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from caip_responses.loop.agent_loop import AgentLoop
from caip_responses.loop.tool_executor import ToolExecutor
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider
from caip_responses.tool_handlers.registry import BuiltinToolRegistry
from caip_responses.tool_handlers.web_search import WebSearchHandler


class _MockProvider(BaseProvider):
    """Provider that returns pre-configured responses in sequence."""

    def __init__(self, responses: list[Response]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self._requests: list[CreateResponseRequest] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    def supports_tool(self, tool_type: str) -> bool:
        return tool_type == "function"

    def supports_reasoning(self) -> bool:
        return False

    async def create_response(self, request: CreateResponseRequest) -> Response:
        self._requests.append(request)
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return self._responses[-1]

    async def create_response_stream(
        self, request: CreateResponseRequest
    ) -> AsyncIterator[StreamEvent]:
        resp = await self.create_response(request)
        yield StreamEvent(
            type="response.created",
            sequence_number=0,
            response={"id": resp.id, "model": resp.model},
        )

        for i, item in enumerate(resp.output):
            if isinstance(item, dict):
                item_type = item.get("type")
            else:
                item_type = getattr(item, "type", None)
                item = item.model_dump()

            yield StreamEvent(
                type="response.output_item.added",
                sequence_number=i + 1,
                output_index=i,
                item=item,
            )

            if item_type == "function_call":
                args = item.get("arguments", "{}")
                yield StreamEvent(
                    type="response.function_call_arguments.delta",
                    sequence_number=i + 2,
                    output_index=i,
                    delta=args,
                )
                yield StreamEvent(
                    type="response.function_call_arguments.done",
                    sequence_number=i + 3,
                    output_index=i,
                    delta=args,
                )

            elif item_type == "message":
                content = item.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        yield StreamEvent(
                            type="response.output_text.delta",
                            sequence_number=i + 2,
                            output_index=i,
                            delta=block.get("text", ""),
                        )

            yield StreamEvent(
                type="response.output_item.done",
                sequence_number=i + 10,
                output_index=i,
            )

        yield StreamEvent(
            type="response.completed",
            sequence_number=100,
            response={"id": resp.id, "model": resp.model, "status": "completed"},
        )


def _text_response(text: str) -> Response:
    return Response(
        id="resp_1",
        model="mock-model",
        output=[{
            "type": "message",
            "id": "item_1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
            "status": "completed",
        }],
    )


def _fc_response(name: str, args: dict, call_id: str = "fc_1") -> Response:
    return Response(
        id="resp_2",
        model="mock-model",
        output=[{
            "type": "function_call",
            "id": "item_2",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(args),
        }],
    )


class TestTwoTierDispatchNonStreaming:
    @pytest.mark.asyncio
    async def test_builtin_handler_intercepts_synthetic_call(self):
        """Synthetic _builtin_web_search_query call is handled by WebSearchHandler."""
        search_results = [
            {"title": "Result 1", "url": "https://r1.com", "snippet": "First result"}
        ]

        async def mock_search(query: str, num_results: int) -> list[dict]:
            return search_results

        registry = BuiltinToolRegistry()
        registry.register(WebSearchHandler(search_callback=mock_search))

        # Step 1: model calls the synthetic search function
        fc_resp = _fc_response(
            "_builtin_web_search_query",
            {"query": "latest news"},
            call_id="fc_ws_1",
        )
        # Step 2: model returns text after seeing search results
        text_resp = _text_response("Here are the latest results.")

        provider = _MockProvider([fc_resp, text_resp])
        executor = ToolExecutor()  # no user handlers
        loop = AgentLoop(provider, executor, builtin_registry=registry)

        request = CreateResponseRequest(model="mock-model", input="Search for news")
        response = await loop.run(request)

        assert response.output_text == "Here are the latest results."
        assert provider._call_count == 2

        # Verify the second request had the search results injected
        second_req = provider._requests[1]
        input_items = second_req.input
        # Should contain function_call_output with search results
        output_items = [
            i for i in input_items
            if isinstance(i, dict) and i.get("type") == "function_call_output"
        ]
        assert len(output_items) == 1
        assert output_items[0]["call_id"] == "fc_ws_1"
        result_data = json.loads(output_items[0]["output"])
        assert result_data["query"] == "latest news"

    @pytest.mark.asyncio
    async def test_user_handler_for_regular_function(self):
        """Regular function calls go to the user's ToolExecutor."""
        fc_resp = _fc_response("get_weather", {"city": "NYC"}, call_id="fc_w_1")
        text_resp = _text_response("NYC is 68F.")

        provider = _MockProvider([fc_resp, text_resp])

        async def weather_handler(name: str, args: dict) -> str:
            return json.dumps({"temp": 68, "unit": "F"})

        executor = ToolExecutor(handlers={"get_weather": weather_handler})
        registry = BuiltinToolRegistry()
        loop = AgentLoop(provider, executor, builtin_registry=registry)

        request = CreateResponseRequest(model="mock-model", input="Weather in NYC?")
        response = await loop.run(request)

        assert response.output_text == "NYC is 68F."
        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_mixed_builtin_and_user_calls(self):
        """Model calls both a builtin and a user function in sequence."""
        async def mock_search(query: str, num_results: int) -> list[dict]:
            return [{"title": "News", "url": "https://n.com", "snippet": "Breaking"}]

        registry = BuiltinToolRegistry()
        registry.register(WebSearchHandler(search_callback=mock_search))

        # Step 1: model calls web search
        fc_search = _fc_response(
            "_builtin_web_search_query",
            {"query": "weather NYC"},
            call_id="fc_s1",
        )
        # Step 2: model calls user function
        fc_weather = _fc_response("get_weather", {"city": "NYC"}, call_id="fc_w1")
        # Step 3: model gives final text
        text_resp = _text_response("Based on search and API: NYC is 70F.")

        provider = _MockProvider([fc_search, fc_weather, text_resp])

        async def weather_handler(name: str, args: dict) -> str:
            return json.dumps({"temp": 70})

        executor = ToolExecutor(handlers={"get_weather": weather_handler})
        loop = AgentLoop(provider, executor, builtin_registry=registry)

        request = CreateResponseRequest(model="mock-model", input="What's the weather?")
        response = await loop.run(request)

        assert response.output_text == "Based on search and API: NYC is 70F."
        assert provider._call_count == 3

    @pytest.mark.asyncio
    async def test_builtin_with_no_registry(self):
        """Without a registry, all calls go to the user executor."""
        fc_resp = _fc_response("_builtin_web_search_query", {"query": "test"})
        text_resp = _text_response("Done.")

        provider = _MockProvider([fc_resp, text_resp])

        async def catch_all(name: str, args: dict) -> str:
            return json.dumps({"handled_by": "user"})

        executor = ToolExecutor()
        executor.set_default_handler(catch_all)
        # No builtin_registry passed
        loop = AgentLoop(provider, executor)

        request = CreateResponseRequest(model="mock-model", input="test")
        response = await loop.run(request)
        assert response.output_text == "Done."


class TestTwoTierDispatchStreaming:
    @pytest.mark.asyncio
    async def test_builtin_handler_in_stream(self):
        """Streaming: synthetic function call handled by builtin handler."""
        async def mock_search(query: str, num_results: int) -> list[dict]:
            return [{"title": "R1", "url": "https://r1.com", "snippet": "S1"}]

        registry = BuiltinToolRegistry()
        registry.register(WebSearchHandler(search_callback=mock_search))

        fc_resp = _fc_response(
            "_builtin_web_search_query",
            {"query": "streaming test"},
            call_id="fc_stream_1",
        )
        text_resp = _text_response("Streamed result.")

        provider = _MockProvider([fc_resp, text_resp])
        executor = ToolExecutor()
        loop = AgentLoop(provider, executor, builtin_registry=registry)

        request = CreateResponseRequest(model="mock-model", input="Stream search")
        events = []
        async for event in loop.run_stream(request):
            events.append(event)

        # Should have events from both steps
        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 2

        text_deltas = [e for e in events if e.type == "response.output_text.delta"]
        assert any("Streamed" in (e.delta or "") for e in text_deltas)
