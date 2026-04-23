from __future__ import annotations

from typing import Any

import pytest

from caip_responses.tool_handlers.base import BuiltinToolHandler


class _ConcreteHandler(BuiltinToolHandler):
    """Minimal concrete handler for testing the ABC."""

    def tool_type(self) -> str:
        return "test_tool"

    def to_function_tools(self, tool_config: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": self._make_fn_name("do_stuff"),
                "description": "Test function",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        return '{"result": "ok"}'


class TestBuiltinToolHandler:
    def test_make_fn_name(self):
        handler = _ConcreteHandler()
        assert handler._make_fn_name("run") == "_builtin_test_tool_run"

    def test_make_fn_name_suffix(self):
        handler = _ConcreteHandler()
        assert handler._make_fn_name("query") == "_builtin_test_tool_query"

    def test_is_synthetic_true(self):
        handler = _ConcreteHandler()
        assert handler.is_synthetic("_builtin_test_tool_run") is True
        assert handler.is_synthetic("_builtin_test_tool_anything") is True

    def test_is_synthetic_false(self):
        handler = _ConcreteHandler()
        assert handler.is_synthetic("get_weather") is False
        assert handler.is_synthetic("_builtin_other_tool_run") is False
        assert handler.is_synthetic("") is False

    def test_tool_type(self):
        handler = _ConcreteHandler()
        assert handler.tool_type() == "test_tool"

    def test_to_function_tools(self):
        handler = _ConcreteHandler()
        tools = handler.to_function_tools({"type": "test_tool"})
        assert len(tools) == 1
        assert tools[0]["name"] == "_builtin_test_tool_do_stuff"
        assert tools[0]["type"] == "function"

    @pytest.mark.asyncio
    async def test_execute(self):
        handler = _ConcreteHandler()
        result = await handler.execute("_builtin_test_tool_do_stuff", {})
        assert result == '{"result": "ok"}'

    def test_system_prompt_addendum_default_none(self):
        handler = _ConcreteHandler()
        assert handler.system_prompt_addendum({"type": "test_tool"}) is None

    def test_to_output_item_default_none(self):
        handler = _ConcreteHandler()
        assert handler.to_output_item("_builtin_test_tool_do_stuff", {}, "{}") is None

    def test_prefix_constant(self):
        assert BuiltinToolHandler._PREFIX == "_builtin_"
