"""Tests for MCPHandler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caip_responses.tool_handlers.mcp import MCPHandler


class TestMCPHandler:
    def test_tool_type(self):
        handler = MCPHandler()
        assert handler.tool_type() == "mcp"

    def test_to_function_tools_generic(self):
        handler = MCPHandler()
        tools = handler.to_function_tools({
            "type": "mcp",
            "server_label": "my_server",
        })
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        assert tool["name"] == "_builtin_mcp_my_server"
        assert "request" in tool["parameters"]["properties"]

    def test_to_function_tools_with_discovered(self):
        """If tools were discovered, each MCP tool becomes a function."""
        handler = MCPHandler()
        handler._server_tools["my_server"] = [
            {
                "type": "function",
                "name": "_builtin_mcp_my_server_get_data",
                "description": "Get data",
                "parameters": {"type": "object", "properties": {}},
                "_mcp_server_label": "my_server",
                "_mcp_server_url": "https://example.com/sse",
                "_mcp_tool_name": "get_data",
            },
        ]
        tools = handler.to_function_tools({
            "type": "mcp", "server_label": "my_server",
        })
        assert len(tools) == 1
        assert tools[0]["name"] == "_builtin_mcp_my_server_get_data"

    def test_is_synthetic(self):
        handler = MCPHandler()
        assert handler.is_synthetic("_builtin_mcp_my_server") is True
        assert handler.is_synthetic("get_weather") is False

    def test_system_prompt_addendum(self):
        handler = MCPHandler()
        addendum = handler.system_prompt_addendum({
            "type": "mcp", "server_label": "my_server",
        })
        assert addendum is not None
        assert "my_server" in addendum
        assert "MCP" in addendum

    @pytest.mark.asyncio
    async def test_execute_no_request_no_mcp(self):
        handler = MCPHandler()
        handler._has_mcp_sdk = False
        result = await handler.execute("_builtin_mcp_srv", {})
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_openai_fallback(self):
        """Falls back to OpenAI when mcp SDK is not available."""
        handler = MCPHandler(openai_api_key="test-key")
        handler._has_mcp_sdk = False

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "MCP result via OpenAI"
        text_block.annotations = []

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        usage = MagicMock()
        usage.input_tokens = 80
        usage.output_tokens = 120
        usage.total_tokens = 200

        response = MagicMock()
        response.output = [message_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        result = await handler.execute(
            "_builtin_mcp_srv", {"request": "fetch user data"}
        )
        data = json.loads(result)

        assert "MCP result via OpenAI" in data["result"]
        assert data["_source"] == "openai_delegation"
        assert data["_usage"]["total_tokens"] == 200

    @pytest.mark.asyncio
    async def test_execute_openai_tracks_metrics(self):
        handler = MCPHandler(openai_api_key="test-key")
        handler._has_mcp_sdk = False

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "Done"
        text_block.annotations = []

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        usage = MagicMock()
        usage.input_tokens = 50
        usage.output_tokens = 75
        usage.total_tokens = 125

        response = MagicMock()
        response.output = [message_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        await handler.execute("_builtin_mcp_srv", {"request": "test"})

        assert handler.metrics.total_calls == 1
        assert handler.metrics.total_tokens == 125

    def test_to_output_item(self):
        handler = MCPHandler()
        item = handler.to_output_item(
            "_builtin_mcp_srv",
            {"request": "test"},
            "{}",
        )
        assert item is not None
        assert item["type"] == "mcp_call"
        assert item["status"] == "completed"

    def test_stores_mcp_configs(self):
        handler = MCPHandler()
        config = {"type": "mcp", "server_label": "srv", "server_url": "https://example.com"}
        handler.to_function_tools(config)
        assert config in handler._mcp_configs

    def test_has_mcp_sdk_check(self):
        """_has_mcp_sdk should be True since mcp is installed in test env."""
        handler = MCPHandler()
        assert handler._has_mcp_sdk is True

    def test_resolve_mcp_tool_found(self):
        handler = MCPHandler()
        handler._server_tools["srv"] = [
            {
                "name": "_builtin_mcp_srv_get_data",
                "_mcp_tool_name": "get_data",
                "_mcp_server_url": "https://example.com/sse",
            },
        ]
        tool_name, url = handler._resolve_mcp_tool("_builtin_mcp_srv_get_data")
        assert tool_name == "get_data"
        assert url == "https://example.com/sse"

    def test_resolve_mcp_tool_not_found(self):
        handler = MCPHandler()
        tool_name, url = handler._resolve_mcp_tool("unknown_fn")
        assert tool_name is None
        assert url is None

    def test_find_server_url_from_configs(self):
        handler = MCPHandler()
        handler._mcp_configs = [
            {"type": "mcp", "server_label": "srv", "server_url": "https://example.com/sse"},
        ]
        assert handler._find_server_url_from_configs() == "https://example.com/sse"

    def test_find_server_url_none(self):
        handler = MCPHandler()
        handler._mcp_configs = [{"type": "mcp", "server_label": "srv"}]
        assert handler._find_server_url_from_configs() is None

    @pytest.mark.asyncio
    async def test_execute_direct_mcp_call(self):
        """Direct MCP call when tool is discovered and SDK is available."""
        handler = MCPHandler()
        handler._has_mcp_sdk = True
        handler._server_tools["srv"] = [
            {
                "name": "_builtin_mcp_srv_get_data",
                "_mcp_tool_name": "get_data",
                "_mcp_server_url": "https://example.com/sse",
            },
        ]

        # Mock the direct execution
        with patch.object(handler, "_execute_direct", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = json.dumps({
                "result": "direct result",
                "_source": "direct_mcp",
            })
            result = await handler.execute(
                "_builtin_mcp_srv_get_data", {"param": "value"}
            )
            data = json.loads(result)
            assert data["_source"] == "direct_mcp"
            mock_exec.assert_called_once_with(
                "get_data", "https://example.com/sse", {"param": "value"}
            )

    @pytest.mark.asyncio
    async def test_error_when_no_sdk_and_no_openai(self):
        """Returns error when neither MCP SDK nor OpenAI is available."""
        handler = MCPHandler()
        handler._has_mcp_sdk = False
        result = await handler.execute(
            "_builtin_mcp_srv", {"request": "test"}
        )
        data = json.loads(result)
        assert "error" in data
        assert "mcp SDK" in data["error"]
