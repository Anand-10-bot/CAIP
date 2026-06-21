from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from caip_responses.loop.agent_loop import AgentLoop
from caip_responses.loop.tool_executor import ToolExecutor
from caip_responses.models.errors import MaxStepsExceededError
from caip_responses.models.request import CreateResponseRequest
from caip_responses.models.response import Response
from caip_responses.models.streaming import StreamEvent
from caip_responses.providers.base import BaseProvider


class _MockProvider(BaseProvider):
    """Provider that returns pre-configured responses in sequence."""

    def __init__(self, responses: list[Response]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    def supports_tool(self, tool_type: str) -> bool:
        return True

    def supports_reasoning(self) -> bool:
        return False

    async def create_response(self, request: CreateResponseRequest) -> Response:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return self._responses[-1]

    async def create_response_stream(self, request: CreateResponseRequest) -> AsyncIterator[StreamEvent]:
        resp = await self.create_response(request)
        # Emit minimal stream events that represent the response
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


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_no_function_calls_returns_immediately(self):
        provider = _MockProvider([_text_response("Hello")])

        async def handler(name: str, args: dict) -> str:
            return "never called"

        executor = ToolExecutor(handlers={"test": handler})
        loop = AgentLoop(provider, executor)

        request = CreateResponseRequest(model="mock-model", input="Hi")
        response = await loop.run(request)

        assert response.output_text == "Hello"
        assert provider._call_count == 1

    @pytest.mark.asyncio
    async def test_single_tool_call_loop(self):
        """Model calls a function, gets the result, then responds with text."""
        fc_resp = _fc_response("get_weather", {"city": "SF"})
        text_resp = _text_response("The weather in SF is 72F.")
        provider = _MockProvider([fc_resp, text_resp])

        async def weather_handler(name: str, args: dict) -> str:
            return json.dumps({"temp": 72})

        executor = ToolExecutor(handlers={"get_weather": weather_handler})
        loop = AgentLoop(provider, executor)

        request = CreateResponseRequest(model="mock-model", input="What's the weather?")
        response = await loop.run(request)

        assert response.output_text == "The weather in SF is 72F."
        assert provider._call_count == 2

    @pytest.mark.asyncio
    async def test_multi_step_tool_calls(self):
        """Model calls tools twice before giving a final text answer."""
        fc1 = _fc_response("lookup", {"q": "a"}, call_id="fc_1")
        fc2 = _fc_response("lookup", {"q": "b"}, call_id="fc_2")
        text = _text_response("Done!")
        provider = _MockProvider([fc1, fc2, text])

        async def handler(name: str, args: dict) -> str:
            return json.dumps({"found": True})

        executor = ToolExecutor(handlers={"lookup": handler})
        loop = AgentLoop(provider, executor)

        request = CreateResponseRequest(model="mock-model", input="Search twice")
        response = await loop.run(request)

        assert response.output_text == "Done!"
        assert provider._call_count == 3

    @pytest.mark.asyncio
    async def test_builtin_output_items_surfaced(self):
        """Builtin handler calls surface as native output items (e.g.
        mcp_call) prepended to the final response for API parity."""
        from caip_responses.tool_handlers.base import BuiltinToolHandler
        from caip_responses.tool_handlers.registry import BuiltinToolRegistry

        class _StubBuiltin(BuiltinToolHandler):
            def tool_type(self) -> str:
                return "mcp"

            def to_function_tools(self, tool_config):
                return []

            async def execute(self, name: str, arguments: dict) -> str:
                return json.dumps({"result": "rolled 7"})

            def to_output_item(self, name, arguments, result):
                return {
                    "type": "mcp_call",
                    "id": "mcp_1",
                    "name": name,
                    "output": result,
                    "status": "completed",
                }

        fc_resp = _fc_response("_builtin_mcp_dmcp_roll", {"sides": 6})
        text_resp = _text_response("You rolled a 7.")
        provider = _MockProvider([fc_resp, text_resp])

        registry = BuiltinToolRegistry()
        registry.register(_StubBuiltin())
        loop = AgentLoop(provider, ToolExecutor(), builtin_registry=registry)

        request = CreateResponseRequest(model="mock-model", input="Roll a die")
        response = await loop.run(request)

        assert response.output_text == "You rolled a 7."
        mcp_calls = [
            item for item in response.output
            if (item.get("type") if isinstance(item, dict)
                else getattr(item, "type", None)) == "mcp_call"
        ]
        assert len(mcp_calls) == 1
        assert mcp_calls[0]["output"] == json.dumps({"result": "rolled 7"})

    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self):
        """If the model keeps calling tools, we eventually raise."""
        # Always returns a function call
        fc = _fc_response("loop_forever", {})
        provider = _MockProvider([fc] * 5)

        async def handler(name: str, args: dict) -> str:
            return "{}"

        executor = ToolExecutor(handlers={"loop_forever": handler})
        loop = AgentLoop(provider, executor, max_steps=3)

        request = CreateResponseRequest(model="mock-model", input="Loop")
        with pytest.raises(MaxStepsExceededError):
            await loop.run(request)

    @pytest.mark.asyncio
    async def test_stream_no_function_calls(self):
        provider = _MockProvider([_text_response("Hi!")])
        executor = ToolExecutor()
        loop = AgentLoop(provider, executor)

        request = CreateResponseRequest(model="mock-model", input="Hello")
        events = []
        async for event in loop.run_stream(request):
            events.append(event)

        # Should have response.created + item events + response.completed
        assert any(e.type == "response.created" for e in events)
        assert any(e.type == "response.completed" for e in events)
        text_deltas = [e for e in events if e.type == "response.output_text.delta"]
        assert len(text_deltas) >= 1
        assert text_deltas[0].delta == "Hi!"

    @pytest.mark.asyncio
    async def test_stream_with_tool_call(self):
        fc_resp = _fc_response("get_weather", {"city": "Delhi"})
        text_resp = _text_response("It's hot in Delhi.")
        provider = _MockProvider([fc_resp, text_resp])

        async def handler(name: str, args: dict) -> str:
            return json.dumps({"temp": 40})

        executor = ToolExecutor(handlers={"get_weather": handler})
        loop = AgentLoop(provider, executor)

        request = CreateResponseRequest(model="mock-model", input="Weather in Delhi?")
        events = []
        async for event in loop.run_stream(request):
            events.append(event)

        # Should see events from both steps
        completed_events = [e for e in events if e.type == "response.completed"]
        assert len(completed_events) == 2  # one per step

        # Final text should be present
        text_deltas = [e for e in events if e.type == "response.output_text.delta"]
        assert any("hot" in (e.delta or "") for e in text_deltas)
