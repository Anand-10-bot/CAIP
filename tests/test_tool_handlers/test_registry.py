from __future__ import annotations

from typing import Any

from caip_responses.tool_handlers.base import BuiltinToolHandler
from caip_responses.tool_handlers.registry import BuiltinToolRegistry


class _FakeHandler(BuiltinToolHandler):
    """Fake handler for testing the registry."""

    def __init__(self, ttype: str) -> None:
        self._ttype = ttype

    def tool_type(self) -> str:
        return self._ttype

    def to_function_tools(self, tool_config: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": self._make_fn_name("run"),
                "description": f"Fake {self._ttype}",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        return "{}"

    def system_prompt_addendum(self, tool_config: dict[str, Any]) -> str | None:
        return f"You have {self._ttype}."


class TestBuiltinToolRegistry:
    def test_register_and_get(self):
        reg = BuiltinToolRegistry()
        handler = _FakeHandler("web_search")
        reg.register(handler)
        assert reg.get("web_search") is handler

    def test_get_unregistered(self):
        reg = BuiltinToolRegistry()
        assert reg.get("nonexistent") is None

    def test_can_handle(self):
        reg = BuiltinToolRegistry()
        reg.register(_FakeHandler("shell"))
        assert reg.can_handle("shell") is True
        assert reg.can_handle("mcp") is False

    def test_resolve_for_function(self):
        reg = BuiltinToolRegistry()
        handler = _FakeHandler("web_search")
        reg.register(handler)
        assert reg.resolve_for_function("_builtin_web_search_run") is handler
        assert reg.resolve_for_function("get_weather") is None

    def test_registered_types(self):
        reg = BuiltinToolRegistry()
        reg.register(_FakeHandler("web_search"))
        reg.register(_FakeHandler("shell"))
        assert sorted(reg.registered_types) == ["shell", "web_search"]

    def test_bool(self):
        reg = BuiltinToolRegistry()
        assert not reg  # empty
        reg.register(_FakeHandler("web_search"))
        assert reg  # non-empty

    def test_preprocess_tools_passthrough_for_supported(self):
        """Tools the provider natively supports should pass through unchanged."""
        reg = BuiltinToolRegistry()
        reg.register(_FakeHandler("web_search"))

        tools = [
            {"type": "function", "name": "get_weather"},
            {"type": "web_search", "search_context_size": "medium"},
        ]
        # Provider supports both function and web_search
        processed, addenda, configs = reg.preprocess_tools(
            tools, {"function", "web_search"}
        )

        assert len(processed) == 2
        assert processed[0] == tools[0]
        assert processed[1] == tools[1]
        assert addenda == []
        assert configs == {}

    def test_preprocess_tools_replaces_unsupported(self):
        """Unsupported tools should be replaced with synthetic functions."""
        reg = BuiltinToolRegistry()
        reg.register(_FakeHandler("web_search"))

        tools = [
            {"type": "function", "name": "get_weather"},
            {"type": "web_search", "search_context_size": "medium"},
        ]
        # Provider only supports function
        processed, addenda, configs = reg.preprocess_tools(tools, {"function"})

        # function tool passes through, web_search replaced with synthetic
        assert len(processed) == 2
        assert processed[0] == tools[0]
        assert processed[1]["type"] == "function"
        assert processed[1]["name"] == "_builtin_web_search_run"
        assert len(addenda) == 1
        assert "web_search" in addenda[0]
        assert "web_search" in configs

    def test_preprocess_tools_unhandled_passthrough(self):
        """Tools with no handler and not supported should pass through."""
        reg = BuiltinToolRegistry()
        # No MCP handler registered

        tools = [{"type": "mcp", "server_url": "http://localhost"}]
        processed, addenda, configs = reg.preprocess_tools(tools, {"function"})

        assert len(processed) == 1
        assert processed[0] == tools[0]
        assert addenda == []

    def test_preprocess_tools_multiple_handlers(self):
        """Multiple unsupported tools should each be replaced."""
        reg = BuiltinToolRegistry()
        reg.register(_FakeHandler("web_search"))
        reg.register(_FakeHandler("shell"))

        tools = [
            {"type": "web_search"},
            {"type": "shell"},
            {"type": "function", "name": "my_fn"},
        ]
        processed, addenda, configs = reg.preprocess_tools(tools, {"function"})

        # function passes through, web_search and shell replaced
        assert len(processed) == 3
        fn_names = [t.get("name") for t in processed]
        assert "_builtin_web_search_run" in fn_names
        assert "_builtin_shell_run" in fn_names
        assert "my_fn" in fn_names
        assert len(addenda) == 2
        assert "web_search" in configs
        assert "shell" in configs

    def test_preprocess_no_addenda_when_handler_returns_none(self):
        """If system_prompt_addendum returns None, no addendum is added."""

        class _NoAddendumHandler(_FakeHandler):
            def system_prompt_addendum(self, tool_config: dict[str, Any]) -> str | None:
                return None

        reg = BuiltinToolRegistry()
        reg.register(_NoAddendumHandler("code_interpreter"))

        tools = [{"type": "code_interpreter"}]
        processed, addenda, configs = reg.preprocess_tools(tools, {"function"})

        assert len(processed) == 1
        assert processed[0]["name"] == "_builtin_code_interpreter_run"
        assert addenda == []
