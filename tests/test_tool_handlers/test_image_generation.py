"""Tests for ImageGenerationHandler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from caip_responses.tool_handlers.image_generation import ImageGenerationHandler


class TestImageGenerationHandler:
    def test_tool_type(self):
        handler = ImageGenerationHandler()
        assert handler.tool_type() == "image_generation"

    def test_to_function_tools(self):
        handler = ImageGenerationHandler()
        tools = handler.to_function_tools({"type": "image_generation"})
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        assert tool["name"] == "_builtin_image_generation_create"
        assert "prompt" in tool["parameters"]["properties"]

    def test_is_synthetic(self):
        handler = ImageGenerationHandler()
        assert handler.is_synthetic("_builtin_image_generation_create") is True
        assert handler.is_synthetic("generate_image") is False

    def test_system_prompt_addendum(self):
        handler = ImageGenerationHandler()
        addendum = handler.system_prompt_addendum({"type": "image_generation"})
        assert addendum is not None
        assert "image" in addendum.lower()

    @pytest.mark.asyncio
    async def test_execute_no_prompt(self):
        handler = ImageGenerationHandler()
        result = await handler.execute("_builtin_image_generation_create", {})
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_no_api_key(self):
        handler = ImageGenerationHandler()
        result = await handler.execute(
            "_builtin_image_generation_create", {"prompt": "a cat"}
        )
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_delegates_to_openai(self):
        handler = ImageGenerationHandler(openai_api_key="test-key")

        ig_item = MagicMock()
        ig_item.type = "image_generation_call"
        ig_item.result = "data:image/png;base64,iVBOR..."
        ig_item.revised_prompt = "A cute orange cat sitting"

        usage = MagicMock()
        usage.input_tokens = 40
        usage.output_tokens = 500
        usage.total_tokens = 540

        response = MagicMock()
        response.output = [ig_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        result = await handler.execute(
            "_builtin_image_generation_create", {"prompt": "a cat"}
        )
        data = json.loads(result)

        assert data["image"] == "data:image/png;base64,iVBOR..."
        assert data["revised_prompt"] == "A cute orange cat sitting"
        assert data["_usage"]["total_tokens"] == 540

    @pytest.mark.asyncio
    async def test_tracks_metrics(self):
        handler = ImageGenerationHandler(openai_api_key="test-key")

        ig_item = MagicMock()
        ig_item.type = "image_generation_call"
        ig_item.result = "url"
        ig_item.revised_prompt = None

        usage = MagicMock()
        usage.input_tokens = 20
        usage.output_tokens = 300
        usage.total_tokens = 320

        response = MagicMock()
        response.output = [ig_item]
        response.usage = usage

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)
        handler._openai_client = mock_client

        await handler.execute(
            "_builtin_image_generation_create", {"prompt": "a dog"}
        )
        assert handler.metrics.total_calls == 1
        assert handler.metrics.total_tokens == 320

    def test_to_output_item(self):
        handler = ImageGenerationHandler()
        result_str = json.dumps({
            "image": "data:image/png;base64,...",
            "revised_prompt": "A cat",
        })
        item = handler.to_output_item(
            "_builtin_image_generation_create",
            {"prompt": "a cat"},
            result_str,
        )
        assert item is not None
        assert item["type"] == "image_generation_call"
        assert item["result"] == "data:image/png;base64,..."
        assert item["revised_prompt"] == "A cat"
