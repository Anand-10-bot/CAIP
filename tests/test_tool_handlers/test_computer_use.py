"""Tests for ComputerUseHandler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from caip_responses.tool_handlers.computer_use import ComputerUseHandler


class TestComputerUseHandler:
    def test_tool_type(self):
        handler = ComputerUseHandler()
        assert handler.tool_type() == "computer_use"

    def test_to_function_tools(self):
        handler = ComputerUseHandler()
        tools = handler.to_function_tools({"type": "computer_use"})
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        assert tool["name"] == "_builtin_computer_use_action"
        assert "instruction" in tool["parameters"]["properties"]

    def test_is_synthetic(self):
        handler = ComputerUseHandler()
        assert handler.is_synthetic("_builtin_computer_use_action") is True
        assert handler.is_synthetic("click_button") is False

    def test_system_prompt_addendum(self):
        handler = ComputerUseHandler()
        addendum = handler.system_prompt_addendum({"type": "computer_use"})
        assert addendum is not None
        assert "computer" in addendum.lower()

    @pytest.mark.asyncio
    async def test_execute_no_instruction(self):
        handler = ComputerUseHandler()
        result = await handler.execute("_builtin_computer_use_action", {})
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_no_api_key(self):
        handler = ComputerUseHandler()
        result = await handler.execute(
            "_builtin_computer_use_action",
            {"instruction": "click submit"},
        )
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_delegates_to_openai(self):
        handler = ComputerUseHandler(openai_api_key="test-key")

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "Clicked the submit button."
        text_block.annotations = []

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 150
        usage.total_tokens = 250

        response = MagicMock()
        response.output = [message_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        result = await handler.execute(
            "_builtin_computer_use_action",
            {"instruction": "click submit button"},
        )
        data = json.loads(result)

        assert "Clicked" in data["result"]
        assert data["_usage"]["total_tokens"] == 250

    @pytest.mark.asyncio
    async def test_execute_passes_display_config(self):
        handler = ComputerUseHandler(openai_api_key="test-key")
        handler.to_function_tools({
            "type": "computer_use",
            "display_width": 1920,
            "display_height": 1080,
        })

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "Done"
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
            "_builtin_computer_use_action",
            {"instruction": "screenshot"},
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        tool_def = call_kwargs["tools"][0]
        assert tool_def["display_width"] == 1920
        assert tool_def["display_height"] == 1080

    @pytest.mark.asyncio
    async def test_tracks_metrics(self):
        handler = ComputerUseHandler(openai_api_key="test-key")

        text_block = MagicMock()
        text_block.type = "output_text"
        text_block.text = "Done"
        text_block.annotations = []

        message_item = MagicMock()
        message_item.type = "message"
        message_item.content = [text_block]

        usage = MagicMock()
        usage.input_tokens = 30
        usage.output_tokens = 50
        usage.total_tokens = 80

        response = MagicMock()
        response.output = [message_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        await handler.execute(
            "_builtin_computer_use_action", {"instruction": "test"}
        )
        assert handler.metrics.total_calls == 1
        assert handler.metrics.total_tokens == 80

    def test_to_output_item(self):
        handler = ComputerUseHandler()
        item = handler.to_output_item(
            "_builtin_computer_use_action",
            {"instruction": "click"},
            "{}",
        )
        assert item is not None
        assert item["type"] == "computer_call"
        assert item["status"] == "completed"
