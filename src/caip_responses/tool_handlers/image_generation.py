"""Image generation tool handler.

Delegates image_generation calls to Azure OpenAI, which handles
DALL-E image generation natively.
"""

from __future__ import annotations

import json
from typing import Any

from caip_responses.tool_handlers.base import BuiltinToolHandler
from caip_responses.tool_handlers.openai_delegator import (
    DelegatedToolMetrics,
    OpenAIDelegatorMixin,
)
from caip_responses.utils.id_gen import generate_item_id


class ImageGenerationHandler(BuiltinToolHandler, OpenAIDelegatorMixin):
    """Client-side image generation tool handler for non-OpenAI providers.

    Delegates to Azure OpenAI's built-in image_generation tool.

    Usage::

        handler = ImageGenerationHandler(
            openai_api_key="sk-...",
            openai_base_url="https://your-resource.openai.azure.com",
        )
        registry.register(handler)
    """

    def __init__(
        self,
        *,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_model: str = "gpt-4.1",
    ) -> None:
        BuiltinToolHandler.__init__(self)
        OpenAIDelegatorMixin.__init__(
            self,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
        )

    @property
    def metrics(self) -> DelegatedToolMetrics:
        return self._delegated_metrics

    def tool_type(self) -> str:
        return "image_generation"

    def to_function_tools(
        self, tool_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": self._make_fn_name("create"),
                "description": (
                    "Generate an image from a text description. "
                    "Returns the generated image URL or base64 data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed description of the image to generate.",
                        },
                    },
                    "required": ["prompt"],
                },
            },
        ]

    async def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> str:
        prompt = arguments.get("prompt", "")
        if not prompt:
            return json.dumps({"error": "No prompt provided"})

        if not self._openai_api_key:
            return json.dumps({
                "error": "image_generation tool requires openai_api_key for delegation",
            })

        client = self._get_openai_client()

        response = await client.responses.create(
            model=self._openai_model,
            input=f"Generate an image: {prompt}",
            tools=[{"type": "image_generation"}],
        )

        usage = self._record_usage(response, "image_generation")

        # Extract image generation results
        image_data = None
        revised_prompt = None
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "image_generation_call":
                image_data = getattr(item, "result", None)
                revised_prompt = getattr(item, "revised_prompt", None)

        text = self._extract_text_from_response(response)

        result: dict[str, Any] = {}
        if image_data:
            result["image"] = image_data
        if revised_prompt:
            result["revised_prompt"] = revised_prompt
        if text:
            result["description"] = text
        result["_usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        }
        return json.dumps(result)

    def to_output_item(
        self,
        name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> dict[str, Any] | None:
        # Parse the result to extract image data
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            data = {}

        item: dict[str, Any] = {
            "type": "image_generation_call",
            "id": generate_item_id(),
            "status": "completed",
            "result": data.get("image"),
        }
        if data.get("revised_prompt"):
            item["revised_prompt"] = data["revised_prompt"]

        last = self.last_delegated_usage
        if last:
            item["_delegated_usage"] = {
                "input_tokens": last.input_tokens,
                "output_tokens": last.output_tokens,
                "total_tokens": last.total_tokens,
                "model": last.model,
                "provider": last.provider,
            }
        return item

    def system_prompt_addendum(
        self, tool_config: dict[str, Any]
    ) -> str | None:
        return (
            "You can generate images by calling the "
            f"'{self._make_fn_name('create')}' function with a detailed "
            "text description of the desired image."
        )
