from __future__ import annotations

import json

import pytest

from caip_responses.loop.tool_executor import ToolExecutor


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_execute_registered_handler(self):
        async def weather_handler(name: str, args: dict) -> str:
            return json.dumps({"temp": 72, "city": args.get("city", "")})

        executor = ToolExecutor(handlers={"get_weather": weather_handler})
        result = await executor.execute(
            call_id="fc_1",
            name="get_weather",
            arguments='{"city": "SF"}',
        )

        assert result["type"] == "function_call_output"
        assert result["call_id"] == "fc_1"
        output = json.loads(result["output"])
        assert output["temp"] == 72
        assert output["city"] == "SF"

    @pytest.mark.asyncio
    async def test_execute_unregistered_returns_error(self):
        executor = ToolExecutor()
        result = await executor.execute(
            call_id="fc_1",
            name="unknown_func",
            arguments="{}",
        )

        output = json.loads(result["output"])
        assert "error" in output
        assert "No handler" in output["error"]

    @pytest.mark.asyncio
    async def test_execute_default_handler(self):
        async def default(name: str, args: dict) -> str:
            return f"handled {name}"

        executor = ToolExecutor()
        executor.set_default_handler(default)

        result = await executor.execute(
            call_id="fc_1",
            name="anything",
            arguments="{}",
        )
        assert result["output"] == "handled anything"

    @pytest.mark.asyncio
    async def test_execute_handler_exception(self):
        async def bad_handler(name: str, args: dict) -> str:
            raise ValueError("something broke")

        executor = ToolExecutor(handlers={"bad": bad_handler})
        result = await executor.execute(
            call_id="fc_1",
            name="bad",
            arguments="{}",
        )

        output = json.loads(result["output"])
        assert "error" in output
        assert "something broke" in output["error"]

    @pytest.mark.asyncio
    async def test_execute_invalid_json_arguments(self):
        async def handler(name: str, args: dict) -> str:
            return json.dumps(args)

        executor = ToolExecutor(handlers={"test": handler})
        result = await executor.execute(
            call_id="fc_1",
            name="test",
            arguments="not-json",
        )

        # Should not crash — args should be empty dict
        output = json.loads(result["output"])
        assert output == {}

    @pytest.mark.asyncio
    async def test_execute_many(self):
        async def handler(name: str, args: dict) -> str:
            return json.dumps({"fn": name, **args})

        executor = ToolExecutor(handlers={"a": handler, "b": handler})
        results = await executor.execute_many([
            {"call_id": "fc_1", "name": "a", "arguments": '{"x": 1}'},
            {"call_id": "fc_2", "name": "b", "arguments": '{"y": 2}'},
        ])

        assert len(results) == 2
        assert results[0]["call_id"] == "fc_1"
        assert results[1]["call_id"] == "fc_2"
        assert json.loads(results[0]["output"])["fn"] == "a"
        assert json.loads(results[1]["output"])["fn"] == "b"

    @pytest.mark.asyncio
    async def test_register_handler(self):
        executor = ToolExecutor()

        async def handler(name: str, args: dict) -> str:
            return "ok"

        executor.register("test", handler)
        result = await executor.execute(
            call_id="fc_1",
            name="test",
            arguments="{}",
        )
        assert result["output"] == "ok"
