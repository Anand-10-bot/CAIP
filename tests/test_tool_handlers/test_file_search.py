"""Tests for FileSearchHandler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from caip_responses.tool_handlers.file_search import FileSearchHandler


class TestFileSearchHandler:
    def test_tool_type(self):
        handler = FileSearchHandler()
        assert handler.tool_type() == "file_search"

    def test_to_function_tools(self):
        handler = FileSearchHandler()
        tools = handler.to_function_tools({"type": "file_search"})
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        assert tool["name"] == "_builtin_file_search_query"
        assert "query" in tool["parameters"]["properties"]

    def test_is_synthetic(self):
        handler = FileSearchHandler()
        assert handler.is_synthetic("_builtin_file_search_query") is True
        assert handler.is_synthetic("my_search") is False

    @pytest.mark.asyncio
    async def test_execute_no_query(self):
        handler = FileSearchHandler()
        result = await handler.execute("_builtin_file_search_query", {})
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_no_api_key(self):
        handler = FileSearchHandler()
        result = await handler.execute(
            "_builtin_file_search_query", {"query": "test"}
        )
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_delegates_to_openai(self):
        handler = FileSearchHandler(openai_api_key="test-key")

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "Found in document: relevant passage"
        text_block.annotations = []

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        usage = MagicMock()
        usage.input_tokens = 60
        usage.output_tokens = 90
        usage.total_tokens = 150

        response = MagicMock()
        response.output = [message_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        result = await handler.execute(
            "_builtin_file_search_query", {"query": "find config"}
        )
        data = json.loads(result)

        assert "relevant passage" in data["result"]
        assert data["_usage"]["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_execute_passes_vector_store_ids(self):
        handler = FileSearchHandler(openai_api_key="test-key")
        handler.to_function_tools({
            "type": "file_search",
            "vector_store_ids": ["vs_123"],
        })

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "Result"
        text_block.annotations = []

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 20
        usage.total_tokens = 30

        response = MagicMock()
        response.output = [message_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        await handler.execute(
            "_builtin_file_search_query", {"query": "test"}
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        tool_def = call_kwargs["tools"][0]
        assert tool_def["type"] == "file_search"
        assert tool_def["vector_store_ids"] == ["vs_123"]

    @pytest.mark.asyncio
    async def test_tracks_metrics(self):
        handler = FileSearchHandler(openai_api_key="test-key")

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "R"
        text_block.annotations = []

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        usage = MagicMock()
        usage.input_tokens = 30
        usage.output_tokens = 40
        usage.total_tokens = 70

        response = MagicMock()
        response.output = [message_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        await handler.execute("_builtin_file_search_query", {"query": "q"})
        assert handler.metrics.total_calls == 1
        assert handler.metrics.total_tokens == 70

    def test_to_output_item(self):
        handler = FileSearchHandler()
        item = handler.to_output_item(
            "_builtin_file_search_query",
            {"query": "test"},
            "{}",
        )
        assert item is not None
        assert item["type"] == "file_search_call"
        assert item["status"] == "completed"
        assert item["queries"] == ["test"]
